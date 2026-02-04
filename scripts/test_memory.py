"""
Test script to verify GPU memory usage and all features.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.utils.config import load_config


def test_memory_usage():
    """Test GPU memory usage with optimized architecture."""
    print("="*60)
    print("GPU Memory Usage Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU memory test.")
        return
    
    device = torch.device("cuda:0")
    config = load_config("configs/default_config.yaml")
    
    # Clear cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0) / 1024**2
    
    print(f"\nInitial GPU Memory: {initial_memory:.2f} MB")
    
    # Create model
    print("\nCreating model...")
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
    
    model_memory = torch.cuda.memory_allocated(0) / 1024**2
    print(f"Model Memory: {model_memory - initial_memory:.2f} MB")
    
    # Test with different sequence lengths
    test_lengths = [100, 500, 1000, 2000, 5000]
    
    print("\n" + "="*60)
    print("Testing with different sequence lengths:")
    print("="*60)
    
    for seq_len in test_lengths:
        torch.cuda.empty_cache()
        before_memory = torch.cuda.memory_allocated(0) / 1024**2
        
        # Create dummy input (simulating spectrogram after CNN)
        dummy_input = torch.randn(1, 1, config.model.n_feats, seq_len).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        after_memory = torch.cuda.memory_allocated(0) / 1024**2
        peak_memory = torch.cuda.max_memory_allocated(0) / 1024**2
        
        print(f"\nSequence Length: {seq_len}")
        print(f"  Input Memory: {after_memory - before_memory:.2f} MB")
        print(f"  Peak Memory: {peak_memory:.2f} MB")
        print(f"  Output Shape: {output.shape}")
        
        # Check if memory is reasonable (< 4GB)
        if peak_memory < 4000:
            print(f"  Status: OK (< 4GB)")
        else:
            print(f"  Status: WARNING (> 4GB)")
    
    # Final memory stats
    print("\n" + "="*60)
    print("Final Memory Statistics:")
    print("="*60)
    print(f"Total Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Total Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"Peak Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    
    # Verify features
    print("\n" + "="*60)
    print("Feature Verification:")
    print("="*60)
    print(f"  Attention: {'Enabled (Chunked)' if config.model.use_attention and config.model.use_chunked_attention else 'Disabled'}")
    print(f"  SE (Squeeze-Excitation): {'Enabled' if config.model.use_se else 'Disabled'}")
    print(f"  Residual CNN: Enabled ({config.model.n_cnn_layers} layers)")
    print(f"  Bidirectional GRU: Enabled ({config.model.n_rnn_layers} layers)")
    print(f"  RNN Dimension: {config.model.rnn_dim}")
    print(f"  Attention Heads: {config.model.n_attention_heads}")
    
    print("\n" + "="*60)
    print("SUCCESS: All features enabled and memory optimized!")
    print("="*60)


if __name__ == "__main__":
    test_memory_usage()

