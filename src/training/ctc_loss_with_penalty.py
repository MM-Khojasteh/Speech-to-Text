"""
CTC Loss with blank token penalty to prevent blank token bias.

The penalty uses a tiered adaptive schedule: for each sample, the highest
tier whose threshold is exceeded by the average blank probability supplies
``probability ** exponent * multiplier``. The tuned tier matrix is part of
the project's private configuration matrix and is injected through the
environment (STT_BLANK_PENALTY_WEIGHT / STT_BLANK_PENALTY_TIERS, see
.env.example); the in-code default below is a deliberately conservative
single-tier profile.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# Conservative public fallback: a single soft tier. The tuned escalating
# matrix is supplied via environment configuration (see .env.example).
DEFAULT_TIER_CONFIG: List[Tuple[float, float, float]] = [(0.0, 1.0, 1.0)]


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
        zero_infinity: bool = False,
        tier_config: Optional[List[Tuple[float, float, float]]] = None
    ):
        """
        Initialize CTC Loss with blank penalty.

        Args:
            blank: Index of blank token
            blank_penalty_weight: Weight for blank penalty (0.0 = no penalty, higher = more penalty)
            reduction: Reduction method ('mean', 'sum', or 'none')
            zero_infinity: Whether to zero infinite losses
            tier_config: Penalty tier matrix as an ascending list of
                (threshold, exponent, multiplier) tuples. For each sample,
                the highest tier whose threshold is exceeded by the average
                blank probability supplies
                ``(probability ** exponent) * multiplier``. Defaults to a
                conservative single-tier profile; the tuned matrix is
                provided through environment configuration.
        """
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
        self.blank = blank
        self.blank_penalty_weight = blank_penalty_weight

        tiers = list(tier_config) if tier_config is not None else list(DEFAULT_TIER_CONFIG)
        if not tiers:
            raise ValueError("tier_config must contain at least one tier")
        for lower, upper in zip(tiers, tiers[1:]):
            if upper[0] <= lower[0]:
                raise ValueError(
                    "tier_config thresholds must be strictly ascending: "
                    f"{tiers!r}"
                )
        self.tier_config = tiers
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

            # Vectorized adaptive penalty calculation.
            # The tier matrix is a list of (threshold, exponent, multiplier)
            # tuples with ascending thresholds; iterating in ascending order
            # means the highest tier exceeded by the average blank probability
            # determines the penalty: (prob ** exponent) * multiplier.
            penalties = torch.zeros_like(avg_blank_probs)
            for threshold, exponent, multiplier in self.tier_config:
                penalties = torch.where(
                    avg_blank_probs > threshold,
                    (avg_blank_probs ** exponent) * multiplier,
                    penalties
                )

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

