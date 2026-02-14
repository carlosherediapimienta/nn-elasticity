import numpy as np
import torch
from .statistics_calculator import StatisticsCalculator
from .knot_generator import KnotGenerator


class SplineBuilder:
    """
    Build spline configuration from training data.
    
    Delegation:
    - StatisticsCalculator: calculate mean/std
    - KnotGenerator: generate knots
    
    Public API: build_from_data().
    """
    
    def __init__(
        self,
        stats_calc: StatisticsCalculator | None = None,
        knot_gen: KnotGenerator | None = None
    ):
        """
        Args:
            stats_calc: custom statistics calculator (optional)
            knot_gen: custom knot generator (optional)
        """
        self.stats_calc = stats_calc or StatisticsCalculator()
        self.knot_gen = knot_gen or KnotGenerator()
    
    def build_from_data(
        self,
        data: np.ndarray,
        n_knots: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95,
        eps: float = 1e-6
    ) -> dict:
        """
        Build spline configuration from training data.
        
        Args:
            data: array with price data (e.g.: log_price_per_liter)
            n_knots: number of knots to generate
            q_min: minimum quantile for knots
            q_max: maximum quantile for knots
            eps: epsilon for std calculation
        
        Returns:
            dict with:
                'knots': tensor with positions of knots
                'mean': mean of the data
                'std': standard deviation of the data
        """
        # Calculate statistics
        stats = self.stats_calc.compute_stats(data, eps=eps)
        
        # Generate knots
        knots = self.knot_gen.generate_from_quantiles(
            data,
            n_knots=n_knots,
            q_min=q_min,
            q_max=q_max
        )
        
        return {
            "knots": knots,
            "mean": stats["mean"],
            "std": stats["std"]
        }