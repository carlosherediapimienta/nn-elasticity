import torch
import torch.nn as nn


class FitLoss:
    """
    Calculate the fit loss (Huber loss) between predictions and targets.
    Public API: run().
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        """
        Args:
            delta: threshold for Huber loss
            reduction: 'mean', 'sum', or 'none'
        """
        self.huber = nn.HuberLoss(delta=delta, reduction=reduction)
    
    def run(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate Huber loss between prediction and target.
        
        Args:
            y_hat: (B,) predictions
            y_true: (B,) true values
        
        Returns:
            loss: scalar (if reduction='mean') or (B,) if reduction='none'
        """
        return self.huber(y_hat, y_true)