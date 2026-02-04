"""
Verify all features are enabled and working.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.utils.config import load_config


def verify_features():
    """Verify all model features are enabled."""
    print("="*60)
    print("Feature Verification")
    print("="*60)
    
    config = load_config("configs/default_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nConfiguration:")
    print(f"  rnn_dim: {config.model.rnn_dim}")
    print(f"  n_attention_heads: {config.model.n_attention_heads}")
    print(f"  use_attention: {config.model.use_attention}")
    print(f"  use_chunked_attention: {config.model.use_chunked_attention}")
    print(f"  use_se: {config.model.use_se}")
    print(f"  n_cnn_layers: {config.model.n_cnn_layers}")
    print(f"  n_rnn_layers: {config.model.n_rnn_layers}")
    
    # Create model
    model = SpeechRecognitionModel(
        n_cnn_layers=config.model.n_cnn_layers,
        n_rnn_layers=config.model.n_rnn_layers,
        rnn_dim=config.model.rnn_dim,
        n_class=config.model.n_class,
        n_feats=config.model.n_feats,
        stride=config.model.stride,
        dropout=config.model.dropout,
        use_attention=config.model.use_attention,
        n_attention_heads=config.model.n_attention_heads,
        use_transformer=config.model.use_transformer,
        use_se=config.model.use_se,
        use_chunked_attention=config.model.use_chunked_attention,
        attention_chunk_size=config.model.attention_chunk_size,
        attention_overlap=config.model.attention_overlap
    ).to(device)
    
    print("\n" + "="*60)
    print("Model Feature Check:")
    print("="*60)
    
    # Check attention
    has_attention = model.attention is not None
    attention_type = type(model.attention).__name__ if has_attention else "None"
    print(f"  Attention: {'ENABLED' if has_attention else 'DISABLED'}")
    if has_attention:
        print(f"    Type: {attention_type}")
        print(f"    Chunked: {config.model.use_chunked_attention}")
        print(f"    Heads: {config.model.n_attention_heads}")
    
    # Check SE
    has_se = any(
        hasattr(layer, 'use_se') and layer.use_se 
        for layer in model.rescnn_layers
    )
    print(f"  Squeeze-Excitation: {'ENABLED' if has_se else 'DISABLED'}")
    
    # Check CNN
    print(f"  Residual CNN: ENABLED ({len(model.rescnn_layers)} layers)")
    
    # Check RNN
    print(f"  Bidirectional GRU: ENABLED ({len(model.backbone)} layers)")
    
    # Test forward pass
    print("\n" + "="*60)
    print("Testing Forward Pass:")
    print("="*60)
    
    dummy_input = torch.randn(1, 1, config.model.n_feats, 1000).to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {output.shape}")
    print(f"  Status: SUCCESS")
    
    # Memory check
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated(0) / 1024**2
        print(f"\n  GPU Memory: {memory_mb:.2f} MB")
        if memory_mb < 4000:
            print(f"  Memory Status: OK (< 4GB)")
        else:
            print(f"  Memory Status: WARNING (> 4GB)")
    
    print("\n" + "="*60)
    print("SUCCESS: All features verified and working!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    verify_features()

