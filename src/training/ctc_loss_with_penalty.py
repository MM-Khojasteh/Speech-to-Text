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
        self._debug_counter = 0
    
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
            T, N = blank_probs.shape

            # Convert input_lengths to tensor if needed
            if isinstance(input_lengths, (list, tuple)):
                lengths = torch.tensor(input_lengths, device=blank_probs.device, dtype=torch.long)
            else:
                lengths = input_lengths.long() if input_lengths.dim() > 0 else input_lengths.unsqueeze(0).long()

            # Create mask for valid positions (non-padded)
            # mask[i, j] = 1 if position i is valid for sample j
            mask = torch.arange(T, device=blank_probs.device).unsqueeze(1) < lengths.unsqueeze(0)  # (T, N)

            # Compute per-sample average blank probability
            # Sum over valid positions and divide by sequence length
            masked_blank_probs = blank_probs * mask.float()
            avg_blank_probs = masked_blank_probs.sum(dim=0) / lengths.clamp(min=1).float()  # (N,)

            # Vectorized adaptive penalty calculation
            # Define penalty multipliers and exponents for different probability ranges
            # Penalty = (prob ^ exp) * weight * multiplier
            penalties = torch.zeros_like(avg_blank_probs)

            # Very high blank prob (>80%): prob^4 * 20x
            mask_vhigh = avg_blank_probs > 0.8
            penalties = torch.where(mask_vhigh, (avg_blank_probs ** 4) * 20.0, penalties)

            # High blank prob (60-80%): prob^4 * 15x
            mask_high = (avg_blank_probs > 0.6) & ~mask_vhigh
            penalties = torch.where(mask_high, (avg_blank_probs ** 4) * 15.0, penalties)

            # Moderate-high blank prob (40-60%): prob^3 * 10x
            mask_mhigh = (avg_blank_probs > 0.4) & ~mask_high & ~mask_vhigh
            penalties = torch.where(mask_mhigh, (avg_blank_probs ** 3) * 10.0, penalties)

            # Moderate blank prob (20-40%): prob^2 * 8x
            mask_mod = (avg_blank_probs > 0.2) & ~mask_mhigh & ~mask_high & ~mask_vhigh
            penalties = torch.where(mask_mod, (avg_blank_probs ** 2) * 8.0, penalties)

            # Low-moderate blank prob (10-20%): prob^2 * 5x
            mask_lowmod = (avg_blank_probs > 0.1) & ~mask_mod & ~mask_mhigh & ~mask_high & ~mask_vhigh
            penalties = torch.where(mask_lowmod, (avg_blank_probs ** 2) * 5.0, penalties)

            # Low blank prob (5-10%): prob^2 * 2x
            mask_low = (avg_blank_probs > 0.05) & ~mask_lowmod & ~mask_mod & ~mask_mhigh & ~mask_high & ~mask_vhigh
            penalties = torch.where(mask_low, (avg_blank_probs ** 2) * 2.0, penalties)

            # Very low blank prob (<5%): prob * 0.2x
            mask_vlow = ~mask_low & ~mask_lowmod & ~mask_mod & ~mask_mhigh & ~mask_high & ~mask_vhigh
            penalties = torch.where(mask_vlow, avg_blank_probs * 0.2, penalties)

            # Apply weight and compute mean across batch
            blank_penalty = (penalties * self.blank_penalty_weight).mean()

            # Combine CTC loss with blank penalty
            total_loss = ctc_loss + blank_penalty

            # Debug: Log penalty values (only occasionally to avoid spam)
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:  # Log every 100 batches
                avg_blank_prob_batch = torch.exp(log_probs[:, :, self.blank]).mean().item()
                print(f"  [Debug] CTC Loss: {ctc_loss.item():.4f}, Blank Penalty: {blank_penalty.item():.4f}, "
                      f"Total: {total_loss.item():.4f}, Avg Blank Prob: {avg_blank_prob_batch:.4f}")
        else:
            total_loss = ctc_loss
        
        return total_loss

