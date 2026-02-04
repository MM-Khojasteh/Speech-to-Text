"""
Decoding functions for CTC-based speech recognition.
Includes greedy decoder and beam search decoder.
"""

import torch
from typing import List, Optional
from collections import defaultdict


def greedy_decoder(
    output_probs: torch.Tensor,
    blank_label: int = 28
) -> List[List[int]]:
    """
    Greedy decoder for CTC outputs.
    Decodes the output probabilities to the most likely indices sequence,
    removing blanks and consecutive duplicates.
    
    Args:
        output_probs: Output probabilities/logits of shape (batch, time, n_classes)
                      or (time, batch, n_classes) for CTC
        blank_label: Index of the blank token (default: 28)
        
    Returns:
        List of decoded sequences (list of integer lists)
    """
    # Handle different input shapes
    if output_probs.dim() == 3:
        if output_probs.size(0) < output_probs.size(1):
            # Assume (time, batch, classes) - transpose to (batch, time, classes)
            output_probs = output_probs.transpose(0, 1)
        # Now shape is (batch, time, classes)
        max_probs, indices = torch.max(output_probs, dim=2)
    else:
        raise ValueError(f"Expected 3D tensor, got {output_probs.dim()}D")
    
    decoded_batches = []
    
    for sequence in indices:
        previous = -1
        decoded_sequence = []
        
        for label_index in sequence:
            label_idx = label_index.item()
            # Remove blanks and consecutive duplicates
            if label_idx != previous and label_idx != blank_label:
                decoded_sequence.append(label_idx)
            previous = label_idx
        
        decoded_batches.append(decoded_sequence)
    
    return decoded_batches


def beam_search_decoder(
    output_probs: torch.Tensor,
    beam_width: int = 10,
    blank_label: int = 28,
    blank_penalty: float = 0.0
) -> List[List[int]]:
    """
    Beam search decoder for CTC outputs.
    More accurate than greedy decoding but slower.
    
    Args:
        output_probs: Output log probabilities of shape (batch, time, n_classes)
        beam_width: Width of the beam (number of hypotheses to keep)
        blank_label: Index of the blank token
        blank_penalty: Penalty for blank tokens (negative value encourages blanks)
        
    Returns:
        List of decoded sequences (list of integer lists)
    """
    if output_probs.dim() != 3:
        raise ValueError(f"Expected 3D tensor (batch, time, classes), got {output_probs.dim()}D")
    
    batch_size, time_steps, num_classes = output_probs.size()
    decoded_batches = []
    
    for batch_idx in range(batch_size):
        # Get log probabilities for this batch
        log_probs = output_probs[batch_idx]  # (time, classes)
        
        # Initialize beam: (prefix, last_char, log_prob)
        beam = [([], -1, 0.0)]
        
        for t in range(time_steps):
            new_beam = defaultdict(lambda: (-1, float('-inf')))
            
            for prefix, last_char, log_prob in beam:
                # Get top-k predictions at this time step
                top_k_probs, top_k_indices = torch.topk(
                    log_probs[t], k=min(beam_width, num_classes)
                )
                
                for prob, char_idx in zip(top_k_probs, top_k_indices):
                    char_idx = char_idx.item()
                    new_log_prob = log_prob + prob.item()
                    
                    if char_idx == blank_label:
                        # Blank: keep prefix unchanged
                        key = (tuple(prefix), last_char)
                        if new_log_prob > new_beam[key][1]:
                            new_beam[key] = (last_char, new_log_prob)
                    elif char_idx == last_char:
                        # Same as last: keep prefix unchanged (CTC collapse)
                        key = (tuple(prefix), last_char)
                        if new_log_prob > new_beam[key][1]:
                            new_beam[key] = (last_char, new_log_prob)
                    else:
                        # New character: extend prefix
                        new_prefix = prefix + [char_idx]
                        key = (tuple(new_prefix), char_idx)
                        if new_log_prob > new_beam[key][1]:
                            new_beam[key] = (char_idx, new_log_prob)
            
            # Keep top beam_width hypotheses
            beam = [
                (list(key[0]), last_char, log_prob)
                for key, (last_char, log_prob) in sorted(
                    new_beam.items(), key=lambda x: x[1][1], reverse=True
                )[:beam_width]
            ]
        
        # Get best hypothesis
        best_prefix, _, _ = max(beam, key=lambda x: x[2])
        decoded_batches.append(best_prefix)
    
    return decoded_batches


def pad_sequences(
    sequences: List[List[int]],
    pad_value: int = 0
) -> torch.Tensor:
    """
    Pad sequences to the same length based on the longest sequence.
    
    Args:
        sequences: List of integer sequences
        pad_value: Value used for padding (default: 0)
        
    Returns:
        Padded tensor of shape (batch, max_length)
    """
    if not sequences:
        return torch.tensor([], dtype=torch.long)
    
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    
    for seq in sequences:
        padded_seq = seq + [pad_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
    
    return torch.tensor(padded_sequences, dtype=torch.long)

