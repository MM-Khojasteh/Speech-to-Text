"""
CNN building blocks for speech recognition model.
Includes residual CNN blocks with layer normalization and squeeze-and-excitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayerNorm(nn.Module):
    """
    Layer normalization built for CNN input.
    Handles the (batch, channel, feature, time) format.
    Normalizes over the feature dimension.
    """
    
    def __init__(self, n_feats: int, eps: float = 1e-5):
        """
        Initialize CNN layer normalization.
        
        Args:
            n_feats: Number of feature dimensions
            eps: Small value for numerical stability
        """
        super(CNNLayerNorm, self).__init__()
        self.n_feats = n_feats
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_feats))
        self.bias = nn.Parameter(torch.zeros(n_feats))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch, channel, feature, time)
            
        Returns:
            Normalized tensor of same shape
        """
        # x: (batch, channel, feature, time)
        # Normalize over feature dimension (dim=2)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # Apply learnable parameters (broadcast weight and bias over feature dimension)
        # weight and bias have shape (n_feats,), need to reshape to (1, 1, n_feats, 1)
        weight = self.weight.view(1, 1, self.n_feats, 1)
        bias = self.bias.view(1, 1, self.n_feats, 1)
        x = x * weight + bias
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE block.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super(SqueezeExcitation, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Global average pooling + FC layers
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation.
        
        Args:
            x: Input tensor of shape (batch, channels, feature, time)
            
        Returns:
            Attention-weighted tensor
        """
        # Global average pooling over feature and time dimensions
        b, c, f, t = x.size()
        se = x.mean(dim=(2, 3), keepdim=False)  # (batch, channels)
        
        # Squeeze and excite
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        
        # Reshape and apply
        se = se.view(b, c, 1, 1)
        return x * se


class ResidualCNN(nn.Module):
    """
    Residual CNN block with layer normalization and optional squeeze-and-excitation.
    Inspired by ResNet but with layer norm instead of batch norm.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        dropout: float,
        n_feats: int,
        use_se: bool = False
    ):
        """
        Initialize residual CNN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel: Convolution kernel size
            stride: Convolution stride
            dropout: Dropout probability
            n_feats: Number of feature dimensions (for layer norm)
            use_se: Whether to use squeeze-and-excitation
        """
        super(ResidualCNN, self).__init__()
        
        self.cnn1 = nn.Conv2d(
            in_channels, out_channels, kernel, stride,
            padding=kernel // 2
        )
        self.cnn2 = nn.Conv2d(
            out_channels, out_channels, kernel, stride,
            padding=kernel // 2
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)
        
        # Squeeze-and-excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels)
        
        # Projection for residual connection if channels change
        if in_channels != out_channels:
            self.projection = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride
            )
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (batch, channel, feature, time)
            
        Returns:
            Output tensor of same shape (or different channels if projection used)
        """
        residual = x  # (batch, channel, feature, time)
        
        # First conv block
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        
        # Second conv block
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        
        # Squeeze-and-excitation
        if self.use_se:
            x = self.se(x)
        
        # Residual connection
        if self.projection is not None:
            residual = self.projection(residual)
        x += residual
        
        return x

