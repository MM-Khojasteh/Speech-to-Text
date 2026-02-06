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
                # Higher blank prob -> higher penalty
                penalty = avg_blank_prob * self.blank_penalty_weight
                blank_penalties.append(penalty)
            
            blank_penalty = torch.stack(blank_penalties).mean()
            
            # Combine CTC loss with blank penalty
            total_loss = ctc_loss + blank_penalty
        else:
            total_loss = ctc_loss
        
        return total_loss

