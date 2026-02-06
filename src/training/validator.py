"""
Validation utilities for speech recognition model.
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple

from ..models.speech_recognition import SpeechRecognitionModel
from ..data.text_transform import TextTransform
from ..inference.decoder import greedy_decoder
from ..metrics.asr_metrics import wer, cer, calculate_batch_metrics


class Validator:
    """
    Validator for speech recognition model evaluation.
    """
    
    def __init__(
        self,
        model: SpeechRecognitionModel,
        text_transform: TextTransform,
        device: torch.device,
        decoder_type: str = 'greedy'
    ):
        """
        Initialize validator.
        
        Args:
            model: Speech recognition model
            text_transform: Text transformation utility
            device: Device to run validation on
            decoder_type: Type of decoder ('greedy' or 'beam')
        """
        self.model = model
        self.text_transform = text_transform
        self.device = device
        self.decoder_type = decoder_type
    
    def validate(
        self,
        validation_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run validation on the validation dataset.
        
        Args:
            validation_loader: Data loader for validation set
            criterion: Loss criterion (e.g., CTC loss)
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing validation metrics (loss, wer, cer)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predicted_texts = []
        all_true_texts = []
        
        with torch.no_grad():
            iterator = tqdm(validation_loader, desc="Validating") if verbose else validation_loader
            
            for batch_idx, data in enumerate(iterator):
                # Handle empty batches (can occur with multiprocessing on Windows)
                if data is None:
                    continue
                
                spectrograms, labels, input_lengths, label_lengths = data
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                output = self.model(spectrograms)
                output = torch.nn.functional.log_softmax(output, dim=2)
                output = output.transpose(0, 1)  # CTC expects (time, batch, classes)
                
                # Calculate loss
                loss = criterion(output, labels, input_lengths, label_lengths)
                total_loss += loss.item()
                
                # Decode predictions
                output_for_decoding = output.transpose(0, 1)  # Back to (batch, time, classes)
                decoded_outputs = greedy_decoder(output_for_decoding)
                
                # Convert to text
                predicted_texts = [
                    self.text_transform.int_to_text(seq)
                    for seq in decoded_outputs
                ]
                true_texts = [
                    self.text_transform.int_to_text(label.tolist())
                    for label in labels
                ]
                
                all_predicted_texts.extend(predicted_texts)
                all_true_texts.extend(true_texts)
        
        # Calculate average metrics
        avg_loss = total_loss / len(validation_loader)
        
        # Filter out empty references/hypotheses for metric calculation
        valid_pairs = [
            (ref, hyp) for ref, hyp in zip(all_true_texts, all_predicted_texts)
            if ref.strip() and hyp.strip()
        ]
        
        if len(valid_pairs) == 0:
            metrics = {
                'loss': avg_loss,
                'wer': float('inf'),
                'cer': float('inf')
            }
            if verbose:
                print(f"\nValidation Results:")
                print(f"  Loss: {avg_loss:.6f}")
                print(f"  Warning: No valid reference-hypothesis pairs for metrics")
            return metrics
        
        valid_refs, valid_hyps = zip(*valid_pairs)
        
        # Calculate comprehensive metrics
        batch_metrics = calculate_batch_metrics(
            list(valid_refs),
            list(valid_hyps),
            ignore_case=False
        )
        
        # Combine with loss
        metrics = {
            'loss': avg_loss,
            **batch_metrics
        }
        
        if verbose:
            print(f"\nValidation Results:")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"\n  Error Rates:")
            print(f"    WER:  {metrics.get('wer', float('inf')):.4f}")
            print(f"    CER:  {metrics.get('cer', float('inf')):.4f}")
            print(f"    SER:  {metrics.get('ser', float('inf')):.4f}")
            print(f"\n  Accuracies:")
            print(f"    Word Accuracy:     {metrics.get('word_accuracy', 0.0):.4f}")
            print(f"    Character Accuracy: {metrics.get('character_accuracy', 0.0):.4f}")
            print(f"    Exact Match Rate:  {metrics.get('exact_match_rate', 0.0):.4f}")
            print(f"\n  N-gram Metrics:")
            print(f"    BLEU:    {metrics.get('bleu', 0.0):.4f}")
            print(f"    ROUGE-L: {metrics.get('rouge_l', 0.0):.4f}")
            print(f"\n  Word-level Precision/Recall/F1:")
            print(f"    Precision: {metrics.get('word_precision', 0.0):.4f}")
            print(f"    Recall:    {metrics.get('word_recall', 0.0):.4f}")
            print(f"    F1:        {metrics.get('word_f1', 0.0):.4f}")
            print(f"\n  Character-level Precision/Recall/F1:")
            print(f"    Precision: {metrics.get('char_precision', 0.0):.4f}")
            print(f"    Recall:    {metrics.get('char_recall', 0.0):.4f}")
            print(f"    F1:        {metrics.get('char_f1', 0.0):.4f}")
        
        return metrics

