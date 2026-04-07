import torch
import torch.nn as nn

class FitLoss:
    """
    Demand fit loss based on Huber (smooth L1) loss.
    Huber loss behaves as MSE for residuals smaller than `delta` and as MAE
    for larger ones, making it robust to outliers in log-demand.
    Args:
        delta     (float): threshold between the quadratic and linear regimes.
        reduction (str):   how to aggregate over the batch (mean, sum, none).
                           or "none" (returns per-sample losses).
    Public API:
        run(y_hat, y_true) -> torch.Tensor
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        self.delta     = float(delta) # Threshold between the quadratic and linear regimes.
        self.reduction = reduction # How to aggregate over the batch (mean, sum, none).
        self.huber     = nn.HuberLoss(delta=self.delta, reduction=self.reduction) # Huber loss.
    
    def run(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        return self.huber(y_hat, y_true)