import numpy as np
import torch


class KnotGenerator:
    """
    Generates knots for splines based on quantiles of the data.
    Public API: generate_from_quantiles().
    """
    
    def generate_from_quantiles(
        self,
        data: np.ndarray,
        n_knots: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95
    ) -> torch.Tensor:
        """
        Generate knots evenly spaced in quantiles.
        
        Args:
            data: array with training data
            n_knots: number of knots to generate
            q_min: minimum quantile (default: 5%)
            q_max: maximum quantile (default: 95%)
        
        Returns:
            tensor with positions of knots
        """
        quantiles = np.linspace(q_min, q_max, n_knots)
        knot_values = np.quantile(data, quantiles)
        return torch.tensor(knot_values, dtype=torch.float32)
    
    def generate_uniform(
        self,
        x_min: float,
        x_max: float,
        n_knots: int = 16
    ) -> torch.Tensor:
        """
        Generate knots evenly spaced in range.
        
        Args:
            x_min: minimum value
            x_max: maximum value
            n_knots: number of knots
        
        Returns:
            tensor with positions of knots
        """
        knot_values = np.linspace(x_min, x_max, n_knots)
        return torch.tensor(knot_values, dtype=torch.float32)