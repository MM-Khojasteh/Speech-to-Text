"""
Script to verify GPU availability and test GPU functionality.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.utils.config import load_config


def verify_gpu():
    """Verify GPU availability and test model on GPU."""
    print("="*60)
    print("GPU Verification Script")
    print("="*60)
    
    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Test GPU functionality
        print("\n" + "="*60)
        print("Testing GPU Functionality")
        print("="*60)
        
        device = torch.device("cuda:0")
        print(f"\nUsing device: {device}")
        
        # Create a test tensor on GPU
        print("\n1. Creating test tensor on GPU...")
        test_tensor = torch.randn(10, 10).to(device)
        print(f"   Tensor device: {test_tensor.device}")
        print(f"   Tensor shape: {test_tensor.shape}")
        
        # Perform computation on GPU
        print("\n2. Performing computation on GPU...")
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"   Result device: {result.device}")
        print(f"   Result shape: {result.shape}")
        
        # Test model on GPU
        print("\n3. Testing model on GPU...")
        try:
            config = load_config("configs/default_config.yaml")
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
                use_se=config.model.use_se
            ).to(device)
            
            # Create dummy input
            dummy_input = torch.randn(1, 1, config.model.n_feats, 100).to(device)
            print(f"   Input device: {dummy_input.device}")
            print(f"   Input shape: {dummy_input.shape}")
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            print(f"   Output device: {output.device}")
            print(f"   Output shape: {output.shape}")
            
            # Check GPU memory usage
            print(f"\n4. GPU Memory Usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            
            print("\n" + "="*60)
            print("SUCCESS: GPU verification successful!")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"\nERROR: Error testing model on GPU: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n" + "="*60)
        print("WARNING: CUDA is not available")
        print("="*60)
        print("\nTo enable GPU support:")
        print("1. Ensure you have an NVIDIA GPU with CUDA support")
        print("2. Install CUDA-enabled PyTorch:")
        print("   Visit: https://pytorch.org/get-started/locally/")
        print("   Example: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\nCurrent PyTorch installation appears to be CPU-only.")
        return False


if __name__ == "__main__":
    verify_gpu()

