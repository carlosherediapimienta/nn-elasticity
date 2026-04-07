import torch
import torch.nn.functional as F

class PositivityPenalty:
    """
    Soft constraint that penalizes positive own-price elasticities.
    Economic theory requires that own-price elasticities be negative (a price
    increase reduces demand). This penalty adds a soft version of that constraint
    by penalizing any elasticity estimate that violates it:
        L_pos = mean( ReLU(eps_hat) )
    ReLU(eps_hat) is zero when eps_hat < 0 (correct sign) and equals eps_hat
    when eps_hat > 0 (violation), so only Giffen-good predictions are penalized.
    When `obs_mask` is provided, the mean is computed only over observed
    (non-missing) product-observation pairs.
    Public API:
        run(eps_hat, obs_mask=None) -> torch.Tensor (scalar)
    """
    
    def run(
        self,
        eps_hat: torch.Tensor,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute the penalty only on the own-price elasticities.
        penalty = F.relu(eps_hat)  # (B, n)
        # If the observation mask is provided, we compute the mean only over the observed products.
        if obs_mask is not None:
            # Recall that .clamp(min=1.0) is used to avoid division by zero.
            # It's a limitation for that denom >= 1.
            denom = obs_mask.sum().clamp(min=1.0)
            return (penalty * obs_mask).sum() / denom
        # Otherwise, we compute the mean over all products.
        return penalty.mean()