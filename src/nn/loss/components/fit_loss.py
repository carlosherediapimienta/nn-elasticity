import torch
import torch.nn as nn


class FitLoss:
    """
    Calculate the fit loss (Huber loss) between predictions and targets.
    Public API: run(y_hat, y_true) -> torch.Tensor.
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        self.delta     = float(delta)
        self.reduction = reduction
        self.huber     = nn.HuberLoss(delta=self.delta, reduction=self.reduction)
    
    def run(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        reduction: str | None = None,
    ) -> torch.Tensor:
        red = reduction or self.reduction
        if red == self.huber.reduction:
            return self.huber(y_hat, y_true)
        return nn.HuberLoss(delta=self.delta, reduction=red)(y_hat, y_true)