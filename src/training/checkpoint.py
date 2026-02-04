"""
Model checkpointing utilities for saving and loading model states.
"""

import torch
import os
from typing import Optional, Dict, Any
from pathlib import Path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    filepath: str = "checkpoint.pth",
    is_best: bool = False,
    best_filepath: Optional[str] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        loss: Current loss value
        metrics: Optional dictionary of metrics (e.g., {'wer': 0.5, 'cer': 0.3})
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
        best_filepath: Path to save best model (if None, uses filepath with '_best' suffix)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics or {}
    }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    # Save best model separately if this is the best
    if is_best:
        if best_filepath is None:
            # Generate best model path from checkpoint path
            path = Path(filepath)
            best_filepath = str(path.parent / f"{path.stem}_best{path.suffix}")
        torch.save(checkpoint, best_filepath)
        print(f"Saved best model to {best_filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint information (epoch, loss, metrics, etc.)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.6f}")
    if 'metrics' in checkpoint and checkpoint['metrics']:
        print(f"  Metrics: {checkpoint['metrics']}")
    
    return checkpoint


def resume_training(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None
) -> int:
    """
    Resume training from checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load
        optimizer: Optimizer to load
        device: Device to load on
        
    Returns:
        Starting epoch number (checkpoint epoch + 1)
    """
    checkpoint = load_checkpoint(filepath, model, optimizer, device)
    start_epoch = checkpoint.get('epoch', 0) + 1
    return start_epoch

