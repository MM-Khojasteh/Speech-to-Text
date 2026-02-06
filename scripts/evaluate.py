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
from src.data.dataset import get_data_loaders, get_custom_data_loaders
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
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (for custom dataset: train/val/test, for LibriSpeech: train/test)"
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
    
    # Get data loader based on dataset type
    dataset_type = getattr(config.data, 'dataset_type', 'librispeech').lower()
    include_persian = getattr(config.data, 'include_persian', True)
    
    print("Loading dataset...")
    
    if dataset_type == 'custom':
        print(f"Using custom dataset")
        test_audio_dir = getattr(config.data, 'test_audio_dir', None)
        
        # Load all data loaders
        train_loader, val_loader, test_loader, text_transform = get_custom_data_loaders(
            audio_files_dir=config.data.audio_files_dir,
            transcripts_dir=config.data.transcripts_dir,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels,
            freq_mask_param=config.data.freq_mask_param,
            time_mask_param=config.data.time_mask_param,
            train_split=getattr(config.data, 'train_split', 0.8),
            seed=config.seed,
            include_persian=include_persian,
            test_audio_dir=test_audio_dir
        )
        
        # Select the appropriate loader based on split
        if args.split == "test":
            if test_loader is None:
                raise ValueError(
                    f"Test set not available. Please ensure test_audio_dir is set in config "
                    f"and contains audio files with matching transcripts."
                )
            data_loader = test_loader
            print(f"Evaluating on test set from {test_audio_dir}")
        elif args.split == "val":
            data_loader = val_loader
            print(f"Evaluating on validation set")
        else:  # train
            data_loader = train_loader
            print(f"Evaluating on training set")
    else:
        # LibriSpeech dataset
        print(f"Using LibriSpeech dataset")
        text_transform = TextTransform(include_persian=include_persian)
        
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
        else:  # train
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
    
    # Get vocabulary size from text transform
    vocab_size = text_transform.get_vocab_size()
    
    # Create model
    print("Creating model...")
    # Use vocab_size from TextTransform instead of config (which may be outdated)
    n_class = vocab_size
    print(f"Using n_class={n_class} (from TextTransform vocabulary)")
    
    model = SpeechRecognitionModel(
        n_cnn_layers=config.model.n_cnn_layers,
        n_rnn_layers=config.model.n_rnn_layers,
        rnn_dim=config.model.rnn_dim,
        n_class=n_class,  # Use dynamic vocabulary size
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
    
    # Create criterion (use standard CTC for evaluation, no penalty needed)
    criterion = nn.CTCLoss(blank=text_transform.blank_id).to(device)
    
    # Create validator
    validator = Validator(model, text_transform, device)
    
    # Evaluate
    print("Running evaluation...")
    metrics = validator.validate(data_loader, criterion, verbose=True)
    
    print("\n" + "="*70)
    print("Evaluation Results:")
    print("="*70)
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"\nError Rates:")
    print(f"  WER: {metrics.get('wer', float('inf')):.4f}")
    print(f"  CER: {metrics.get('cer', float('inf')):.4f}")
    print(f"  SER: {metrics.get('ser', float('inf')):.4f}")
    print(f"\nAccuracies:")
    print(f"  Word Accuracy:     {metrics.get('word_accuracy', 0.0):.4f}")
    print(f"  Character Accuracy: {metrics.get('character_accuracy', 0.0):.4f}")
    print(f"  Exact Match Rate:  {metrics.get('exact_match_rate', 0.0):.4f}")
    print(f"\nN-gram Metrics:")
    print(f"  BLEU:    {metrics.get('bleu', 0.0):.4f}")
    print(f"  ROUGE-L: {metrics.get('rouge_l', 0.0):.4f}")
    print(f"\nWord-level Metrics:")
    print(f"  Precision: {metrics.get('word_precision', 0.0):.4f}")
    print(f"  Recall:    {metrics.get('word_recall', 0.0):.4f}")
    print(f"  F1:        {metrics.get('word_f1', 0.0):.4f}")
    print(f"\nCharacter-level Metrics:")
    print(f"  Precision: {metrics.get('char_precision', 0.0):.4f}")
    print(f"  Recall:    {metrics.get('char_recall', 0.0):.4f}")
    print(f"  F1:        {metrics.get('char_f1', 0.0):.4f}")
    print("="*70)


if __name__ == "__main__":
    main()

