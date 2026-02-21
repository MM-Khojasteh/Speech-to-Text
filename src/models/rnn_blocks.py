"""
RNN and Transformer building blocks for speech recognition model.
Includes bidirectional GRU and multi-head self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for sequence modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        batch_size, seq_len, _ = x.size()
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        # Residual connection
        return output + residual


class ChunkedMultiHeadSelfAttention(nn.Module):
    """
    Memory-efficient multi-head self-attention with chunked processing.
    Processes attention in overlapping windows to reduce memory from O(n²) to O(n×w).
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        chunk_size: int = 512,
        overlap: int = 64
    ):
        """
        Initialize chunked multi-head self-attention.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
            chunk_size: Size of each attention chunk (default: 512)
            overlap: Overlap between chunks for smooth transitions (default: 64)
        """
        super(ChunkedMultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through chunked multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask (not used in chunked version)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        batch_size, seq_len, _ = x.size()
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # If sequence is shorter than chunk size, use standard attention
        if seq_len <= self.chunk_size:
            return self._standard_attention(x, residual)
        
        # Process in chunks with overlap
        output_chunks = []
        step_size = self.chunk_size - self.overlap
        
        for start_idx in range(0, seq_len, step_size):
            end_idx = min(start_idx + self.chunk_size, seq_len)
            chunk = x[:, start_idx:end_idx, :]  # (batch, chunk_len, d_model)
            chunk_residual = residual[:, start_idx:end_idx, :]  # Matching residual chunk
            
            # Process chunk with standard attention
            chunk_output = self._standard_attention(chunk, chunk_residual)
            
            # Handle overlap: blend with previous chunk
            if start_idx > 0 and len(output_chunks) > 0:
                # Blend the overlapping region
                overlap_len = min(self.overlap, chunk_output.shape[1], output_chunks[-1].shape[1])
                if overlap_len > 0:
                    # Weighted average in overlap region
                    alpha = torch.linspace(0, 1, overlap_len, device=x.device).view(1, -1, 1)
                    prev_overlap = output_chunks[-1][:, -overlap_len:, :]
                    curr_overlap = chunk_output[:, :overlap_len, :]
                    chunk_output[:, :overlap_len] = (
                        alpha * curr_overlap + (1 - alpha) * prev_overlap
                    )
            
            output_chunks.append(chunk_output)

        # Concatenate all chunks
        output = torch.cat(output_chunks, dim=1)

        # Ensure output matches residual size exactly
        if output.shape[1] != residual.shape[1]:
            if output.shape[1] > residual.shape[1]:
                # Trim excess (due to overlap at the end)
                output = output[:, :residual.shape[1], :]
            else:
                # This shouldn't happen, but pad if needed
                padding = torch.zeros(
                    output.shape[0], residual.shape[1] - output.shape[1], output.shape[2],
                    device=output.device, dtype=output.dtype
                )
                output = torch.cat([output, padding], dim=1)

        # Residual connection
        return output + residual
    
    def _standard_attention(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Standard multi-head attention computation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            residual: Residual tensor for connection
            
        Returns:
            Output tensor of same shape
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        
        return output


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU with layer normalization, residual connection, and dropout.

    Uses a projection layer for the residual if input dim != output dim (hidden_size * 2).
    """

    def __init__(
        self,
        rnn_dim: int,
        hidden_size: int,
        dropout: float,
        batch_first: bool = True
    ):
        """
        Initialize bidirectional GRU.

        Args:
            rnn_dim: Input dimension
            hidden_size: Hidden state size
            dropout: Dropout probability
            batch_first: If True, input/output is (batch, seq, features)
        """
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        output_dim = hidden_size * 2
        self.use_projection = rnn_dim != output_dim
        if self.use_projection:
            self.residual_proj = nn.Linear(rnn_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bidirectional GRU with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, rnn_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_size * 2)
        """
        residual = x

        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)

        # Add residual connection
        if self.use_projection:
            residual = self.residual_proj(residual)
        return x + residual


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize transformer encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
        """
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention
        x = self.attention(x, mask)
        
        # Feed-forward with residual
        residual = x
        x = self.layer_norm(x)
        x = self.ffn(x)
        x = x + residual
        
        return x

