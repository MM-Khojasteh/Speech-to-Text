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
        sample_pairs = []  # Store first few pairs for display
        
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
                decoded_outputs = greedy_decoder(output_for_decoding, blank_label=self.text_transform.blank_id)
                
                # Convert to text
                predicted_texts = []
                for seq in decoded_outputs:
                    text = self.text_transform.int_to_text(seq)
                    # Ensure we have a string, not a list
                    if isinstance(text, list):
                        text = text[0] if text else ''
                    predicted_texts.append(str(text) if text else '')
                
                # Fix: Use label_lengths to slice only the actual label portion (not padding)
                true_texts = []
                for label, label_len in zip(labels, label_lengths):
                    text = self.text_transform.int_to_text(label[:label_len].tolist())
                    # Ensure we have a string, not a list
                    if isinstance(text, list):
                        text = text[0] if text else ''
                    true_texts.append(str(text) if text else '')
                
                # Ensure both lists have the same length
                batch_size = len(true_texts)
                if len(predicted_texts) != batch_size:
                    # Pad or truncate to match
                    if len(predicted_texts) < batch_size:
                        predicted_texts.extend([''] * (batch_size - len(predicted_texts)))
                    else:
                        predicted_texts = predicted_texts[:batch_size]
                
                # Store sample pairs (first batch, up to 5 samples)
                if batch_idx == 0 and len(sample_pairs) < 5:
                    for ref, hyp in zip(true_texts, predicted_texts):
                        if len(sample_pairs) >= 5:
                            break
                        sample_pairs.append((ref, hyp))
                
                all_predicted_texts.extend(predicted_texts)
                all_true_texts.extend(true_texts)
        
        # Calculate average metrics
        avg_loss = total_loss / len(validation_loader)
        
        # Ensure both lists are the same length (safety check)
        min_length = min(len(all_true_texts), len(all_predicted_texts))
        if len(all_true_texts) != len(all_predicted_texts):
            print(f"Warning: Mismatch in list lengths - true_texts: {len(all_true_texts)}, predicted_texts: {len(all_predicted_texts)}")
            all_true_texts = all_true_texts[:min_length]
            all_predicted_texts = all_predicted_texts[:min_length]
        
        # Calculate statistics
        total_pairs = len(all_true_texts)
        empty_refs = sum(1 for ref in all_true_texts if isinstance(ref, str) and not ref.strip())
        empty_hyps = sum(1 for hyp in all_predicted_texts if isinstance(hyp, str) and not hyp.strip())
        non_empty_refs = total_pairs - empty_refs
        non_empty_hyps = total_pairs - empty_hyps
        
        # Filter out empty references/hypotheses for metric calculation
        valid_pairs = [
            (ref, hyp) for ref, hyp in zip(all_true_texts, all_predicted_texts)
            if isinstance(ref, str) and isinstance(hyp, str) and ref.strip() and hyp.strip()
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
                print(f"\n  Statistics:")
                print(f"    Total samples: {total_pairs}")
                print(f"    Empty references: {empty_refs} ({100*empty_refs/max(total_pairs,1):.1f}%)")
                print(f"    Empty predictions: {empty_hyps} ({100*empty_hyps/max(total_pairs,1):.1f}%)")
                print(f"    Valid pairs (both non-empty): {len(valid_pairs)}")
                print(f"\n  Warning: No valid reference-hypothesis pairs for metrics")
                
                # Show sample pairs even if empty
                if sample_pairs:
                    print(f"\n  Sample Predictions (first {len(sample_pairs)}):")
                    for i, (ref, hyp) in enumerate(sample_pairs, 1):
                        ref_str = str(ref) if not isinstance(ref, str) else ref
                        hyp_str = str(hyp) if not isinstance(hyp, str) else hyp
                        ref_display = ref_str[:50] + "..." if len(ref_str) > 50 else ref_str
                        hyp_display = hyp_str[:50] + "..." if len(hyp_str) > 50 else hyp_str
                        ref_status = "(empty)" if not ref_str.strip() else ""
                        hyp_status = "(empty)" if not hyp_str.strip() else ""
                        print(f"    {i}. Reference: {ref_display} {ref_status}")
                        print(f"       Prediction: {hyp_display} {hyp_status}")
                
                # Show sample non-empty references if all predictions are empty
                if empty_hyps == total_pairs and non_empty_refs > 0:
                    print(f"\n  Sample References (what model should learn):")
                    sample_refs = [ref for ref in all_true_texts if isinstance(ref, str) and ref.strip()][:3]
                    for i, ref in enumerate(sample_refs, 1):
                        ref_display = ref[:60] + "..." if len(ref) > 60 else ref
                        print(f"    {i}. {ref_display}")
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
            print(f"\n  Statistics:")
            print(f"    Total samples: {total_pairs}")
            print(f"    Empty references: {empty_refs} ({100*empty_refs/max(total_pairs,1):.1f}%)")
            print(f"    Empty predictions: {empty_hyps} ({100*empty_hyps/max(total_pairs,1):.1f}%)")
            print(f"    Valid pairs (both non-empty): {len(valid_pairs)} ({100*len(valid_pairs)/max(total_pairs,1):.1f}%)")
            
            # Show sample predictions
            if sample_pairs:
                print(f"\n  Sample Predictions (first {min(len(sample_pairs), 5)}):")
                for i, (ref, hyp) in enumerate(sample_pairs[:5], 1):
                    ref_str = str(ref) if not isinstance(ref, str) else ref
                    hyp_str = str(hyp) if not isinstance(hyp, str) else hyp
                    ref_display = ref_str[:60] + "..." if len(ref_str) > 60 else ref_str
                    hyp_display = hyp_str[:60] + "..." if len(hyp_str) > 60 else hyp_str
                    ref_status = " (empty)" if not ref_str.strip() else ""
                    hyp_status = " (empty)" if not hyp_str.strip() else ""
                    print(f"    {i}. Reference: {ref_display}{ref_status}")
                    print(f"       Prediction: {hyp_display}{hyp_status}")
            
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

