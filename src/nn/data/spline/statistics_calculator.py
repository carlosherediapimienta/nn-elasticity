import numpy as np


class StatisticsCalculator:
    """
    Calculates statistics of data for normalization.
    Public API: compute_stats().
    """
    
    def compute_stats(
        self,
        data: np.ndarray,
        eps: float = 1e-6
    ) -> dict[str, float]:
        """
        Calculate mean and standard deviation.
        
        Args:
            data: array of data
            eps: minimum value to avoid division by zero in std
        
        Returns:
            dict with {"mean": float, "std": float}
        """
        mean = float(data.mean())
        std  = float(max(data.std(), 0.05) + eps)
        
        return {
            "mean": mean,
            "std": std
        }