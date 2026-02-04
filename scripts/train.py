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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.speech_recognition import SpeechRecognitionModel
from src.data.dataset import get_data_loaders
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
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
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
    
    # Set device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Create text transform
    text_transform = TextTransform()
    
    # Get data loaders
    print("Loading datasets...")
    train_loader, val_loader = get_data_loaders(
        train_root=config.data.train_root,
        test_root=config.data.test_root,
        train_url=config.data.train_url,
        test_url=config.data.test_url,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
        freq_mask_param=config.data.freq_mask_param,
        time_mask_param=config.data.time_mask_param
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
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
        use_se=config.model.use_se
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, train_loader)
    
    # Create criterion
    criterion = nn.CTCLoss(blank=text_transform.blank_id).to(device)
    
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
        early_stopping_patience=config.training.early_stopping_patience
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

