"""Quick test script to check GPU and test architecture with audio file."""
import torch
import torchaudio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.data.text_transform import TextTransform
from src.data.transforms import get_valid_transforms
from src.inference.decoder import greedy_decoder

# Check GPU
print("="*50)
print("GPU Availability Check")
print("="*50)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
else:
    print("Using CPU")
print(f"PyTorch Version: {torch.__version__}")
print("="*50)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# Create model
print("Creating model...")
text_transform = TextTransform()
model = SpeechRecognitionModel(
    n_cnn_layers=3,
    n_rnn_layers=5,
    rnn_dim=512,
    n_class=29,
    n_feats=128,
    stride=2,
    dropout=0.1,
    use_attention=True,
    n_attention_heads=8,
    use_transformer=False,
    use_se=False
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Model created successfully!\n")

# Load and process audio file
audio_file = "test_audio/test_file_1.wav"
print(f"Loading audio file: {audio_file}")

try:
    # Try loading with soundfile directly as workaround
    import soundfile as sf
    data, sample_rate = sf.read(audio_file)
    waveform = torch.from_numpy(data).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension
    if waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.t()  # Transpose if needed (channels, samples)
    print(f"Audio loaded: shape={waveform.shape}, sample_rate={sample_rate}")
    
    # Create audio transforms
    audio_transforms = get_valid_transforms(sample_rate=16000, n_mels=128)
    
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
        print(f"Resampled to: {sample_rate} Hz")
    
    # Apply transforms
    spec = audio_transforms(waveform)
    print(f"Spectrogram shape: {spec.shape}")
    
    # Prepare for model: spec is (channels, freq, time), need (batch, channels, freq, time)
    if spec.dim() == 3:
        spec = spec.unsqueeze(0)  # Add batch dimension -> (batch, channels, freq, time)
    elif spec.dim() == 2:
        spec = spec.unsqueeze(0).unsqueeze(0)  # (batch, channels, freq, time)
    spec = spec.to(device)
    print(f"Model input shape: {spec.shape}\n")
    
    # Run model inference
    print("Running model inference...")
    model.eval()
    with torch.no_grad():
        output = model(spec)
        output = torch.nn.functional.log_softmax(output, dim=2)
    
    print(f"Model output shape: {output.shape}")
    
    # Decode
    decoded_output = greedy_decoder(output)
    predicted_text = text_transform.int_to_text(decoded_output[0]) if decoded_output else ""
    
    print("\n" + "="*50)
    print("Inference Results")
    print("="*50)
    print(f"Predicted Text: {predicted_text}")
    print("="*50)
    print("\n✓ Architecture test completed successfully!")
    print("Note: This is with an untrained model, so output will be random.")
    print("Train the model first for meaningful predictions.")
    
except FileNotFoundError:
    print(f"ERROR: Audio file not found: {audio_file}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

