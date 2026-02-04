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
from ..metrics.asr_metrics import wer, cer


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
        
        # Calculate WER and CER
        wer_scores = []
        cer_scores = []
        
        for ref, hyp in zip(all_true_texts, all_predicted_texts):
            try:
                wer_scores.append(wer(ref, hyp))
            except ValueError:
                # Skip if reference is empty
                pass
            try:
                cer_scores.append(cer(ref, hyp))
            except ValueError:
                # Skip if reference is empty
                pass
        
        avg_wer = np.mean(wer_scores) if wer_scores else float('inf')
        avg_cer = np.mean(cer_scores) if cer_scores else float('inf')
        
        metrics = {
            'loss': avg_loss,
            'wer': avg_wer,
            'cer': avg_cer
        }
        
        if verbose:
            print(f"\nValidation Results:")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"  WER:  {avg_wer:.4f}")
            print(f"  CER:  {avg_cer:.4f}")
        
        return metrics

