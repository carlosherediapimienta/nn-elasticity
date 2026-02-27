import torch
import torch.nn as nn
from .components import FitLoss, SmoothnessPenalty, PositivityPenalty


class ElasticityLoss(nn.Module):
    """
    Compound loss for elasticity model:
      loss = Huber(y_hat, y) + λ_smooth * mean((d²y/dx²)²) + λ_pos * mean(ReLU(eps_hat))
    
    where:
      - Huber: fit loss
      - smooth: smoothness penalty (promotes smoothness)
      - pos: positivity penalty (promotes negative elasticities)
    
    Delegation:
    - FitLoss: calcula Huber loss
    - SmoothnessPenalty: calculate smoothness penalty
    - PositivityPenalty: calculate positivity penalty
    
    Public API: run().
    """
    
    def __init__(
        self,
        huber_delta: float = 1.0,
        lambda_smooth: float = 0.0,
        lambda_pos: float = 0.0,
        reduction: str = "mean",
        # Optional: dependency injection
        fit_loss: FitLoss | None = None,
        smoothness_penalty: SmoothnessPenalty | None = None,
        positivity_penalty: PositivityPenalty | None = None
    ):
        """
        Args:
            huber_delta: threshold for Huber loss
            lambda_smooth: smoothness penalty weight
            lambda_pos: positivity penalty weight
            reduction: reduction method for Huber loss
            fit_loss: custom fit loss calculator (optional)
            smoothness_penalty: custom smoothness calculator (optional)
            positivity_penalty: custom positivity calculator (optional)
        """
        super().__init__()
        
        # Dependency injection or default creation
        self.fit_loss = fit_loss or FitLoss(delta=huber_delta, reduction=reduction)
        self.smoothness_penalty = smoothness_penalty or SmoothnessPenalty()
        self.positivity_penalty = positivity_penalty or PositivityPenalty()
        
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_pos = float(lambda_pos)

    def run(
        self,
        y_hat: torch.Tensor,    # (B, n)
        y_true: torch.Tensor,   # (B, n)
        eps_hat: torch.Tensor,  # (B, n)
        w: torch.Tensor,        # (B, n, K)
        ddBx: torch.Tensor,     # (B, n, K)
        u: torch.Tensor,        # (B, n_cross, K, K)
        Bx: torch.Tensor,       # (B, n, K)
        IBx: torch.Tensor,      # (B, n, K)
        pairs: torch.Tensor,    # (2, n_cross)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        loss_fit = self.fit_loss.run(y_hat, y_true)

        if self.lambda_smooth > 0.0:
            loss_smooth = self.smoothness_penalty.run(w, ddBx, u, Bx, IBx, pairs)
        else:
            loss_smooth = y_hat.new_tensor(0.0)

        if self.lambda_pos > 0.0:
            loss_pos = self.positivity_penalty.run(eps_hat)
        else:
            loss_pos = y_hat.new_tensor(0.0)

        loss = loss_fit + self.lambda_smooth * loss_smooth + self.lambda_pos * loss_pos

        logs = {
            "loss":       loss.detach(),
            "loss_fit":   loss_fit.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_pos":   loss_pos.detach(),
            "eps_mean":   eps_hat.detach().mean(),
            "eps_p50":    eps_hat.detach().median(),
        }
        return loss, logs
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)