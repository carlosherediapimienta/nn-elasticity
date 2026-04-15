import torch
import torch.nn as nn
from .components import FitLoss, SmoothnessPenalty, PositivityPenalty


class ElasticityLoss(nn.Module):
    """
    Compound loss for elasticity model:
      loss = Huber + Smoothness + Positivity
      where:
        - Huber: fit loss
        - Smoothness: smoothness penalty (promotes smoothness)
        - Positivity: positivity penalty (promotes negative elasticities)
    
    Public API: run().
    """
    
    def __init__(
        self,
        huber_delta: float = 1.0,
        lambda_smooth: float = 0.0,
        lambda_pos: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Args:
            huber_delta: threshold for Huber loss
            lambda_smooth: smoothness penalty weight
            lambda_pos: positivity penalty weight
            reduction: reduction method for Huber loss
        """
        super().__init__()
        
        # Huber loss, smoothness penalty, and positivity penalty.
        self.fit_loss = FitLoss(delta=huber_delta, reduction=reduction)
        self.smoothness_penalty = SmoothnessPenalty()
        self.positivity_penalty = PositivityPenalty()
        
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_pos = float(lambda_pos)

    def run(
        self,
        y_hat: torch.Tensor,     # (B, n)
        y_true: torch.Tensor,    # (B, n)
        eps_hat: torch.Tensor,   # (B, n)
        obs_mask: torch.Tensor,  # (B, n)
        w: torch.Tensor,         # (B, n, K)
        ddBx: torch.Tensor,      # (B, n, K)
        u: torch.Tensor,         # (B, n_cross, K, K)
        Bx: torch.Tensor,        # (B, n, K)
        pairs: torch.Tensor,     # (2, n_cross)
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # 1. Fit loss - only on observed demands - demand fit loss based on Huber (smooth L1) loss.
        mask = obs_mask.bool()
        if mask.any():
            loss_fit = self.fit_loss.run(y_hat[mask], y_true[mask])
        else:
            loss_fit = y_hat.new_tensor(0.0)

        # 2. Smoothness penalty - regularization penalty that discourages highly curved demand curves.
        if self.lambda_smooth > 0.0:
            loss_smooth = self.smoothness_penalty.run(w, ddBx, u, Bx, pairs)
        else:
            loss_smooth = y_hat.new_tensor(0.0)

        # 3. Positivity penalty - only on observed products -
        #  soft constraint that penalizes positive own-price elasticities.   
        if self.lambda_pos > 0.0:
            loss_pos = self.positivity_penalty.run(eps_hat, obs_mask)
        else:
            loss_pos = y_hat.new_tensor(0.0)

        # Compute the total loss.
        loss = loss_fit + self.lambda_smooth * loss_smooth + self.lambda_pos * loss_pos

        # Compute the logs. We detach the tensors to free the memory.
        logs = {
            "loss":        loss.detach(),
            "loss_fit":    loss_fit.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_pos":    loss_pos.detach(),
            "eps_mean":    eps_hat.detach().mean(),
            "eps_p50":     eps_hat.detach().median(),
            "obs_frac":    obs_mask.mean().detach(),
        }
        return loss, logs
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)