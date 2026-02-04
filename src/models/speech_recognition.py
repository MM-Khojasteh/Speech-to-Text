"""
Enhanced Speech Recognition Model inspired by DeepSpeech 2.
Includes CNN feature extraction, optional attention, and RNN/Transformer backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_blocks import ResidualCNN, CNNLayerNorm
from .rnn_blocks import BidirectionalGRU, TransformerEncoderLayer, MultiHeadSelfAttention


class SpeechRecognitionModel(nn.Module):
    """
    Enhanced Speech Recognition Model with attention mechanisms.
    
    Architecture:
    1. CNN feature extraction (with residual blocks)
    2. Optional multi-head self-attention
    3. Bidirectional RNN or Transformer encoder
    4. Linear classifier for CTC output
    """
    
    def __init__(
        self,
        n_cnn_layers: int = 3,
        n_rnn_layers: int = 5,
        rnn_dim: int = 512,
        n_class: int = 29,
        n_feats: int = 128,
        stride: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        n_attention_heads: int = 8,
        use_transformer: bool = False,
        use_se: bool = False
    ):
        """
        Initialize speech recognition model.
        
        Args:
            n_cnn_layers: Number of residual CNN layers
            n_rnn_layers: Number of RNN/Transformer layers
            rnn_dim: RNN hidden dimension
            n_class: Number of output classes (vocab size)
            n_feats: Number of mel spectrogram features
            stride: CNN stride (reduces sequence length)
            dropout: Dropout probability
            use_attention: Whether to use multi-head attention
            n_attention_heads: Number of attention heads
            use_transformer: If True, use Transformer instead of RNN
            use_se: Whether to use squeeze-and-excitation in CNN blocks
        """
        super(SpeechRecognitionModel, self).__init__()
        
        self.n_feats = n_feats // 2  # After initial CNN
        self.rnn_dim = rnn_dim
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        
        # Initial CNN for feature extraction
        self.cnn = nn.Conv2d(
            1, 32, kernel_size=3, stride=stride, padding=3 // 2
        )
        
        # Residual CNN layers
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(
                32, 32, kernel=3, stride=1,
                dropout=dropout, n_feats=self.n_feats, use_se=use_se
            )
            for _ in range(n_cnn_layers)
        ])
        
        # Fully connected layer to project CNN features to RNN dimension
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)
        
        # Optional multi-head self-attention
        if use_attention:
            self.attention = MultiHeadSelfAttention(
                d_model=rnn_dim,
                n_heads=n_attention_heads,
                dropout=dropout
            )
        else:
            self.attention = None
        
        # RNN or Transformer backbone
        if use_transformer:
            # Transformer encoder layers
            self.backbone = nn.Sequential(*[
                TransformerEncoderLayer(
                    d_model=rnn_dim,
                    n_heads=n_attention_heads,
                    d_ff=rnn_dim * 4,
                    dropout=dropout
                )
                for _ in range(n_rnn_layers)
            ])
            # Transformer outputs same dimension
            backbone_output_dim = rnn_dim
        else:
            # Bidirectional GRU layers
            self.backbone = nn.Sequential(*[
                BidirectionalGRU(
                    rnn_dim=rnn_dim if i == 0 else rnn_dim * 2,
                    hidden_size=rnn_dim,
                    dropout=dropout,
                    batch_first=(i == 0)
                )
                for i in range(n_rnn_layers)
            ])
            # BiGRU outputs rnn_dim * 2
            backbone_output_dim = rnn_dim * 2
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(backbone_output_dim, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input spectrogram tensor of shape (batch, 1, freq, time)
            
        Returns:
            Output logits of shape (batch, time, n_class)
        """
        # CNN feature extraction
        x = self.cnn(x)  # (batch, 32, freq, time)
        x = self.rescnn_layers(x)  # (batch, 32, freq, time)
        
        # Reshape for RNN/Transformer
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        
        # Project to RNN dimension
        x = self.fully_connected(x)  # (batch, time, rnn_dim)
        
        # Optional attention
        if self.attention is not None:
            x = self.attention(x)  # (batch, time, rnn_dim)
        
        # RNN or Transformer backbone
        x = self.backbone(x)  # (batch, time, backbone_output_dim)
        
        # Classifier
        x = self.classifier(x)  # (batch, time, n_class)
        
        return x

