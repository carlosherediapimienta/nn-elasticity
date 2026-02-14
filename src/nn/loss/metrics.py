import torch

class RegressionMetrics:
    """
    Calculate standard regression metrics.
    Public API: run().
    """
    
    def run(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Calculate MAE and MSE.
        
        Args:
            y_hat: (B,) predictions
            y_true: (B,) true values
        
        Returns:
            dict with 'mae' and 'mse'
        """
        mae = torch.mean(torch.abs(y_hat - y_true))
        mse = torch.mean((y_hat - y_true) ** 2)
        
        return {
            "mae": mae,
            "mse": mse
        }