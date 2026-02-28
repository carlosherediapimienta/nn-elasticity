import torch
import torch.nn.functional as F

class PositivityPenalty:
    """
    Positivity penalty: mean(ReLU(eps_hat)).
    Promotes negative elasticities (downward sloping demand).
    Public API: run(eps_hat, obs_mask) -> torch.Tensor.
    """
    
    def run(
        self,
        eps_hat: torch.Tensor,
        obs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        penalty = F.relu(eps_hat)  # (B, n)
        if obs_mask is not None:
            denom = obs_mask.sum().clamp(min=1.0)
            return (penalty * obs_mask).sum() / denom
        return penalty.mean()