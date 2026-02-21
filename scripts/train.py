"""
Training script for speech recognition model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from pathlib import Path
import sys
import platform

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.data.dataset import get_data_loaders, get_custom_data_loaders
from src.data.text_transform import TextTransform
from src.training.trainer import Trainer
from src.utils.config import load_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model: nn.Module, config) -> optim.Optimizer:
    """Create optimizer based on config."""
    return optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )


def create_scheduler(optimizer: optim.Optimizer, config, train_loader) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler based on config."""
    scheduler_type = config.training.scheduler_type.lower()
    scheduler_params = config.training.scheduler_params
    
    if scheduler_type == "onecycle":
        # Use max_lr if specified, otherwise default to 10x learning_rate
        max_lr = getattr(config.training, 'max_lr', config.training.learning_rate * 10)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(train_loader),
            epochs=config.training.epochs,
            **scheduler_params
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs * len(train_loader),
            **scheduler_params
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train speech recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Set device with explicit GPU verification
    if config.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            # Clear GPU cache before training
            torch.cuda.empty_cache()
            print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
        print(f"Using device: {device}")
    
    # Create text transform (with Persian support if enabled)
    include_persian = getattr(config.data, 'include_persian', True)
    text_transform = TextTransform(include_persian=include_persian)
    
    # Get vocabulary size dynamically from TextTransform
    vocab_size = text_transform.get_vocab_size()
    print(f"Vocabulary size: {vocab_size} (includes blank token)")
    
    # Get data loaders based on dataset type
    print("Loading datasets...")
    dataset_type = getattr(config.data, 'dataset_type', 'librispeech').lower()
    
    # On Windows, set num_workers to 0 to avoid multiprocessing issues
    num_workers = config.training.num_workers
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"Windows detected: Setting num_workers to 0 (was {num_workers}) to avoid multiprocessing issues")
        num_workers = 0
    
    if dataset_type == 'custom':
        print(f"Using custom dataset from {config.data.audio_files_dir}")
        test_audio_dir = getattr(config.data, 'test_audio_dir', None)
        train_loader, val_loader, test_loader, text_transform = get_custom_data_loaders(
            audio_files_dir=config.data.audio_files_dir,
            transcripts_dir=config.data.transcripts_dir,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
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
        # Update vocab size after loading custom dataset (in case it changed)
        vocab_size = text_transform.get_vocab_size()
        print(f"Vocabulary size after dataset load: {vocab_size}")
        
        if test_loader is not None:
            print(f"Test batches: {len(test_loader)} (from {test_audio_dir})")
        else:
            print("No test set loaded (test_audio_dir not provided or empty)")
    else:
        print(f"Using LibriSpeech dataset")
        train_loader, val_loader = get_data_loaders(
            train_root=config.data.train_root,
            test_root=config.data.test_root,
            train_url=config.data.train_url,
            test_url=config.data.test_url,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
            pin_memory=config.training.pin_memory,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels,
            freq_mask_param=config.data.freq_mask_param,
            time_mask_param=config.data.time_mask_param
        )
    
    print(f"\nDataset Summary:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model with dynamic vocabulary size
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, train_loader)
    
    # Create criterion with blank penalty to prevent blank token bias
    from src.training.ctc_loss_with_penalty import CTCLossWithBlankPenalty
    criterion = CTCLossWithBlankPenalty(
        blank=text_transform.blank_id,
        blank_penalty_weight=30.0  # Very strong penalty weight to prevent blank bias
    ).to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        text_transform=text_transform,
        use_amp=config.training.use_amp,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        save_best=True,
        early_stopping_patience=config.training.early_stopping_patience,
        warmup_steps=getattr(config.training, 'warmup_steps', 0)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.resume_from_checkpoint(args.resume)
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=config.training.epochs,
        start_epoch=start_epoch,
        validate_every=config.training.validate_every,
        save_every=config.training.save_every
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()

