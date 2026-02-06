"""
Training utilities for speech recognition model with advanced features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, Any
import os

from ..models.speech_recognition import SpeechRecognitionModel
from ..training.validator import Validator
from ..training.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """
    Trainer for speech recognition model with AMP, gradient accumulation, and checkpointing.
    """
    
    def __init__(
        self,
        model: SpeechRecognitionModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        device: torch.device,
        text_transform,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Speech recognition model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss criterion
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            text_transform: Text transformation utility
            use_amp: Whether to use Automatic Mixed Precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            save_best: Whether to save best model based on validation loss
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.text_transform = text_transform
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_best = save_best
        self.early_stopping_patience = early_stopping_patience
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Validator
        if val_loader is not None:
            self.validator = Validator(model, text_transform, device)
        else:
            self.validator = None
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_wer = float('inf')
        self.epochs_without_improvement = 0
        
        # TensorBoard writer (optional)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}"
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, data in progress_bar:
            spectrograms, labels, input_lengths, label_lengths = data
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    output = self.model(spectrograms)
                    output = torch.nn.functional.log_softmax(output, dim=2)
                    output = output.transpose(0, 1)  # CTC expects (time, batch, classes)
                    
                    loss = self.criterion(output, labels, input_lengths, label_lengths)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                output = self.model(spectrograms)
                output = torch.nn.functional.log_softmax(output, dim=2)
                output = output.transpose(0, 1)
                
                loss = self.criterion(output, labels, input_lengths, label_lengths)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to TensorBoard
            if self.writer is not None:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item() * self.gradient_accumulation_steps, global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def train(
        self,
        num_epochs: int,
        start_epoch: int = 0,
        validate_every: int = 1,
        save_every: int = 10
    ) -> None:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming)
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
        """
        self.current_epoch = start_epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = None
            if self.val_loader is not None and (epoch + 1) % validate_every == 0:
                val_metrics = self.validator.validate(self.val_loader, self.criterion)
                
                # Check for best model
                if self.save_best and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_wer = val_metrics.get('wer', float('inf'))
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                    )
                    save_checkpoint(
                        self.model, self.optimizer, epoch, val_metrics['loss'],
                        val_metrics, checkpoint_path, is_best=True,
                        best_filepath=os.path.join(self.checkpoint_dir, "best_model.pth")
                    )
                else:
                    self.epochs_without_improvement += 1
                
                # Log validation metrics
                if self.writer is not None:
                    # Core metrics
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/WER', val_metrics.get('wer', float('inf')), epoch)
                    self.writer.add_scalar('Val/CER', val_metrics.get('cer', float('inf')), epoch)
                    
                    # Error rates
                    if 'ser' in val_metrics:
                        self.writer.add_scalar('Val/SER', val_metrics['ser'], epoch)
                    
                    # Accuracies
                    if 'word_accuracy' in val_metrics:
                        self.writer.add_scalar('Val/WordAccuracy', val_metrics['word_accuracy'], epoch)
                    if 'character_accuracy' in val_metrics:
                        self.writer.add_scalar('Val/CharAccuracy', val_metrics['character_accuracy'], epoch)
                    if 'exact_match_rate' in val_metrics:
                        self.writer.add_scalar('Val/ExactMatchRate', val_metrics['exact_match_rate'], epoch)
                    
                    # N-gram metrics
                    if 'bleu' in val_metrics:
                        self.writer.add_scalar('Val/BLEU', val_metrics['bleu'], epoch)
                    if 'rouge_l' in val_metrics:
                        self.writer.add_scalar('Val/ROUGE-L', val_metrics['rouge_l'], epoch)
                    
                    # Word-level PRF
                    if 'word_f1' in val_metrics:
                        self.writer.add_scalar('Val/WordF1', val_metrics['word_f1'], epoch)
                        self.writer.add_scalar('Val/WordPrecision', val_metrics.get('word_precision', 0.0), epoch)
                        self.writer.add_scalar('Val/WordRecall', val_metrics.get('word_recall', 0.0), epoch)
                    
                    # Character-level PRF
                    if 'char_f1' in val_metrics:
                        self.writer.add_scalar('Val/CharF1', val_metrics['char_f1'], epoch)
                        self.writer.add_scalar('Val/CharPrecision', val_metrics.get('char_precision', 0.0), epoch)
                        self.writer.add_scalar('Val/CharRecall', val_metrics.get('char_recall', 0.0), epoch)
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
                )
                metrics = val_metrics if val_metrics else train_metrics
                save_checkpoint(
                    self.model, self.optimizer, epoch, metrics['loss'],
                    metrics, checkpoint_path, is_best=False
                )
            
            # Early stopping
            if (self.early_stopping_patience is not None and
                self.epochs_without_improvement >= self.early_stopping_patience):
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"No improvement for {self.epochs_without_improvement} epochs")
                break
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Starting epoch number
        """
        checkpoint = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('loss', float('inf'))
        
        if 'metrics' in checkpoint:
            self.best_val_wer = checkpoint['metrics'].get('wer', float('inf'))
        
        return start_epoch

