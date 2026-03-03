import numpy as np
import torch


class KnotGenerator:
    """
    Generates knots for splines based on quantiles of the data.
    Public API: generate_from_quantiles().
    """
    
    def generate_from_quantiles(self, data: np.ndarray, n_knots=16, q_min=0.05, q_max=0.95):
        x = np.asarray(data, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("There are no finite values for the spline knots.")

        quantiles = np.linspace(q_min, q_max, n_knots)
        knot_values = np.quantile(x, quantiles)
        return torch.tensor(knot_values, dtype=torch.float32)
    
    def generate_uniform(self, x_min: float, x_max: float, n_knots: int = 16):
        knot_values = np.linspace(x_min, x_max, n_knots)
        return torch.tensor(knot_values, dtype=torch.float32)