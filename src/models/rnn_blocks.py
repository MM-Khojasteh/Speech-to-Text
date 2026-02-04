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


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU with layer normalization and dropout.
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bidirectional GRU.
        
        Args:
            x: Input tensor of shape (batch, seq_len, rnn_dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, rnn_dim * 2)
        """
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


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

