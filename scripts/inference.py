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
    waveform, file_sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if file_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(file_sample_rate, sample_rate)
        waveform = resampler(waveform)
    
    # Apply audio transforms (MelSpectrogram)
    spec = audio_transforms(waveform)
    
    # Handle shape: (1, freq, time) -> (1, 1, freq, time) for batch
    if spec.dim() == 3:
        spec = spec.unsqueeze(0)  # Add batch dimension
    spec = spec.transpose(2, 3)  # (batch, 1, time, freq)
    
    # Move to device
    spec = spec.to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(spec)
        output = torch.nn.functional.log_softmax(output, dim=2)
    
    # Decode output
    decoded_output = greedy_decoder(output)
    
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
    
    # Set device
    if config.device == "cuda" and not torch.cuda.is_available():
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Run inference
    print(f"Processing audio: {args.audio}")
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

