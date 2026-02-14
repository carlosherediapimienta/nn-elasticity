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
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        eps_hat: torch.Tensor,
        w: torch.Tensor,
        ddBx: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Public API. Calculate total loss and logs.
        
        Args:
            y_hat: (B,) predicted demand
            y_true: (B,) true demand
            eps_hat: (B,) predicted elasticities
            w: (B, K) spline weights
            ddBx: (B, K) second derivatives of spline bases
        
        Returns:
            loss: scalar, total loss
            logs: dict with detailed metrics
        """
        # 1. Fit loss (Huber)
        loss_fit = self.fit_loss.run(y_hat, y_true)

        # 2. Smoothness penalty (curvature)
        if self.lambda_smooth > 0.0:
            loss_smooth = self.smoothness_penalty.run(w, ddBx)
        else:
            loss_smooth = y_hat.new_tensor(0.0)

        # 3. Positivity penalty (elasticity)
        if self.lambda_pos > 0.0:
            loss_pos = self.positivity_penalty.run(eps_hat)
        else:
            loss_pos = y_hat.new_tensor(0.0)

        # 4. Total loss
        loss = loss_fit + self.lambda_smooth * loss_smooth + self.lambda_pos * loss_pos
        
        # 5. Logging
        logs = {
            "loss": loss.detach(),
            "loss_fit": loss_fit.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_pos": loss_pos.detach(),
            "eps_mean": eps_hat.detach().mean(),
            "eps_p50": eps_hat.detach().median(),
        }
        
        return loss, logs
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)