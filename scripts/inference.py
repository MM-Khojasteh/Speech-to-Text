"""
Inference script for speech recognition model.
"""

import argparse
import torch
import torchaudio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.data.text_transform import TextTransform
from src.data.transforms import get_valid_transforms
from src.inference.decoder import greedy_decoder
from src.utils.config import load_config
from src.training.checkpoint import load_checkpoint


def speech_to_text_chunked(
    audio_path: str,
    model: SpeechRecognitionModel,
    device: torch.device,
    text_transform: TextTransform,
    audio_transforms,
    sample_rate: int = 16000,
    chunk_duration: float = 30.0,
    overlap_duration: float = 2.0
) -> str:
    """
    Perform speech-to-text conversion on long audio files using chunked processing.
    
    Args:
        audio_path: Path to audio file
        model: Trained speech recognition model
        device: Device to run inference on
        text_transform: Text transformation utility
        audio_transforms: Audio transformation pipeline
        sample_rate: Expected sample rate
        chunk_duration: Duration of each chunk in seconds (default: 30s)
        overlap_duration: Overlap between chunks in seconds (default: 2s)
        
    Returns:
        Predicted text string
    """
    # Load audio file
    try:
        import soundfile as sf
        data, file_sample_rate = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
    except (ImportError, Exception):
        try:
            waveform, file_sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
    
    # Resample if necessary
    if file_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)
    
    # Calculate audio duration
    audio_duration = waveform.shape[-1] / sample_rate
    
    # If audio is short, use standard processing
    if audio_duration <= chunk_duration:
        return speech_to_text(audio_path, model, device, text_transform, audio_transforms, sample_rate)
    
    print(f"Processing long audio ({audio_duration:.1f}s) in chunks of {chunk_duration}s...")
    
    # Calculate chunk sizes
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    all_decoded_outputs = []
    model.eval()
    
    with torch.no_grad():
        for start_idx in range(0, waveform.shape[-1], step_samples):
            end_idx = min(start_idx + chunk_samples, waveform.shape[-1])
            chunk_waveform = waveform[:, start_idx:end_idx]
            
            # Apply audio transforms
            spec = audio_transforms(chunk_waveform)
            
            # Handle shape
            if spec.dim() == 3:
                spec = spec.unsqueeze(0)
            
            spec = spec.to(device)
            
            # Run model inference
            output = model(spec)  # (time, batch, classes)
            output = torch.nn.functional.log_softmax(output, dim=2)

            # Transpose to (batch, time, classes) for decoder
            output = output.transpose(0, 1)

            # Decode output
            decoded_output = greedy_decoder(output, blank_label=text_transform.blank_id)
            if decoded_output and len(decoded_output) > 0:
                all_decoded_outputs.extend(decoded_output[0])
            
            # Clear GPU cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Merge decoded outputs (remove duplicates at boundaries)
    if all_decoded_outputs:
        # Simple deduplication: remove consecutive duplicates
        merged_output = []
        prev = None
        for item in all_decoded_outputs:
            if item != prev:
                merged_output.append(item)
            prev = item
        predicted_text = text_transform.int_to_text(merged_output)
    else:
        predicted_text = ""
    
    return predicted_text


def speech_to_text(
    audio_path: str,
    model: SpeechRecognitionModel,
    device: torch.device,
    text_transform: TextTransform,
    audio_transforms,
    sample_rate: int = 16000
) -> str:
    """
    Perform speech-to-text conversion on an audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained speech recognition model
        device: Device to run inference on
        text_transform: Text transformation utility
        audio_transforms: Audio transformation pipeline
        sample_rate: Expected sample rate
        
    Returns:
        Predicted text string
    """
    # Load audio file
    # Try using soundfile directly for better compatibility
    try:
        import soundfile as sf
        data, file_sample_rate = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
    except (ImportError, Exception):
        # Fallback to torchaudio
        try:
            waveform, file_sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")
    
    # Resample if necessary
    if file_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)
    
    # Apply audio transforms (MelSpectrogram)
    spec = audio_transforms(waveform)
    
    # Handle shape: (1, freq, time) -> (1, 1, freq, time) for batch
    if spec.dim() == 3:
        spec = spec.unsqueeze(0)  # Add batch dimension if needed
    # spec should be (batch, 1, freq, time) - no transpose needed
    
    # Move to device
    spec = spec.to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(spec)  # (time, batch, classes)
        output = torch.nn.functional.log_softmax(output, dim=2)
        # Transpose to (batch, time, classes) for decoder
        output = output.transpose(0, 1)

    # Decode output
    decoded_output = greedy_decoder(output, blank_label=text_transform.blank_id)
    
    # Convert to text
    if decoded_output and len(decoded_output) > 0:
        predicted_text = text_transform.int_to_text(decoded_output[0])
    else:
        predicted_text = ""
    
    return predicted_text


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio file")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device with explicit GPU verification
    if config.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
        print(f"Using device: {device}")
    
    # Create text transform
    text_transform = TextTransform()
    
    # Create audio transforms
    audio_transforms = get_valid_transforms(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels
    )
    
    # Create model
    print("Loading model...")
    # For inference on smaller GPUs, disable attention if sequence is too long
    # Attention can be memory-intensive for long sequences
    use_attention = config.model.use_attention
    if use_attention:
        print("Note: Attention enabled. For very long audio, consider disabling it in config.")
    
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
        use_chunked_attention=getattr(config.model, 'use_chunked_attention', True),
        attention_chunk_size=getattr(config.model, 'attention_chunk_size', 512),
        attention_overlap=getattr(config.model, 'attention_overlap', 64)
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Run inference (use chunked for long audio)
    print(f"Processing audio: {args.audio}")
    
    # Check audio duration to decide on chunked processing
    try:
        import soundfile as sf
        data, _ = sf.read(args.audio)
        audio_duration = len(data) / config.data.sample_rate if hasattr(config.data, 'sample_rate') else len(data) / 16000
    except:
        audio_duration = 0
    
    # Use chunked inference for audio longer than 30 seconds
    if audio_duration > 30.0:
        predicted_text = speech_to_text_chunked(
            args.audio,
            model,
            device,
            text_transform,
            audio_transforms,
            config.data.sample_rate,
            chunk_duration=30.0,
            overlap_duration=2.0
        )
    else:
        predicted_text = speech_to_text(
            args.audio,
            model,
            device,
            text_transform,
            audio_transforms,
            config.data.sample_rate
        )
    
    print("\n" + "="*50)
    print("Predicted Text:")
    print("="*50)
    print(predicted_text)
    print("="*50)


if __name__ == "__main__":
    main()

