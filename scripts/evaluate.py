"""
Evaluation script for speech recognition model.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.data.dataset import get_data_loaders
from src.data.text_transform import TextTransform
from src.training.validator import Validator
from src.utils.config import load_config
from src.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate speech recognition model")
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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate on"
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
    
    # Get data loader
    print("Loading dataset...")
    if args.split == "test":
        _, data_loader = get_data_loaders(
            train_root=config.data.train_root,
            test_root=config.data.test_root,
            train_url=config.data.train_url,
            test_url=config.data.test_url,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels
        )
    else:
        data_loader, _ = get_data_loaders(
            train_root=config.data.train_root,
            test_root=config.data.test_root,
            train_url=config.data.train_url,
            test_url=config.data.test_url,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels
        )
    
    print(f"Evaluation batches: {len(data_loader)}")
    
    # Create model
    print("Creating model...")
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
    
    # Create criterion
    criterion = nn.CTCLoss(blank=text_transform.blank_id).to(device)
    
    # Create validator
    validator = Validator(model, text_transform, device)
    
    # Evaluate
    print("Running evaluation...")
    metrics = validator.validate(data_loader, criterion, verbose=True)
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"WER:  {metrics['wer']:.4f}")
    print(f"CER:  {metrics['cer']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()

