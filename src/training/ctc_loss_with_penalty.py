"""
CTC Loss with blank token penalty to prevent blank token bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWithBlankPenalty(nn.Module):
    """
    CTC Loss wrapper that adds penalty for excessive blank token predictions.
    This helps prevent the model from learning to predict only blank tokens.
    """
    
    def __init__(
        self,
        blank: int,
        blank_penalty_weight: float = 0.1,
        reduction: str = 'mean',
        zero_infinity: bool = False
    ):
        """
        Initialize CTC Loss with blank penalty.
        
        Args:
            blank: Index of blank token
            blank_penalty_weight: Weight for blank penalty (0.0 = no penalty, higher = more penalty)
            reduction: Reduction method ('mean', 'sum', or 'none')
            zero_infinity: Whether to zero infinite losses
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
        self.blank = blank
        self.blank_penalty_weight = blank_penalty_weight
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss with blank penalty.
        
        Args:
            log_probs: Log probabilities of shape (T, N, C) where T=time, N=batch, C=classes
            targets: Target sequences of shape (N*S,) where S=max target length
            input_lengths: Lengths of input sequences
            target_lengths: Lengths of target sequences
            
        Returns:
            Loss value with blank penalty
        """
        # Standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Always calculate blank penalty to see what's happening
        if self.blank_penalty_weight > 0.0:
            # Calculate blank penalty: penalize high average probability of blank token
            # Convert log_probs to probs for blank token
            blank_probs = torch.exp(log_probs[:, :, self.blank])  # (T, N)
            
            # Average blank probability across time steps for each sample
            # Weight by input_lengths to handle variable lengths
            batch_size = blank_probs.shape[1]
            blank_penalties = []
            
            for i in range(batch_size):
                # Handle both list and tensor input_lengths
                if isinstance(input_lengths, (list, tuple)):
                    length = input_lengths[i]
                else:
                    length = input_lengths[i].item() if hasattr(input_lengths[i], 'item') else int(input_lengths[i])
                
                avg_blank_prob = blank_probs[:length, i].mean()
                # Penalty: encourage lower blank probability
                # Use adaptive penalty based on blank probability level
                # Higher blank prob -> exponentially higher penalty
                # Normal blank prob in CTC is typically 0.05-0.15, so we penalize anything above 0.1
                if avg_blank_prob > 0.8:
                    # Very strong penalty for very high blank probability (>80%)
                    penalty = (avg_blank_prob ** 4) * self.blank_penalty_weight * 20.0
                elif avg_blank_prob > 0.6:
                    # Very strong penalty for high blank probability (60-80%)
                    penalty = (avg_blank_prob ** 4) * self.blank_penalty_weight * 15.0
                elif avg_blank_prob > 0.4:
                    # Strong penalty for moderate-high blank probability (40-60%)
                    penalty = (avg_blank_prob ** 3) * self.blank_penalty_weight * 10.0
                elif avg_blank_prob > 0.2:
                    # Moderate-strong penalty for moderate blank probability (20-40%)
                    penalty = (avg_blank_prob ** 2) * self.blank_penalty_weight * 8.0
                elif avg_blank_prob > 0.1:
                    # Moderate-strong penalty for low-moderate blank probability (10-20%)
                    penalty = (avg_blank_prob ** 2) * self.blank_penalty_weight * 5.0
                elif avg_blank_prob > 0.05:
                    # Moderate penalty for low blank probability (5-10%)
                    penalty = (avg_blank_prob ** 2) * self.blank_penalty_weight * 2.0
                else:
                    # Light penalty for very low blank probability (<5% - normal range)
                    penalty = avg_blank_prob * self.blank_penalty_weight * 0.2
                blank_penalties.append(penalty)
            
            blank_penalty = torch.stack(blank_penalties).mean()
            
            # Combine CTC loss with blank penalty
            total_loss = ctc_loss + blank_penalty
            
            # Debug: Log penalty values (only occasionally to avoid spam)
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 100 == 0:  # Log every 100 batches
                avg_blank_prob_batch = torch.exp(log_probs[:, :, self.blank]).mean().item()
                print(f"  [Debug] CTC Loss: {ctc_loss.item():.4f}, Blank Penalty: {blank_penalty.item():.4f}, "
                      f"Total: {total_loss.item():.4f}, Avg Blank Prob: {avg_blank_prob_batch:.4f}")
        else:
            total_loss = ctc_loss
        
        return total_loss

