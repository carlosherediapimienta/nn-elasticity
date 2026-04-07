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
    
    def __init__(self):
        self.stats_calc = StatisticsCalculator() # Statistics calculator
        self.knot_gen = KnotGenerator() # Knot generator
    
    def build_from_data(
        self,
        data: np.ndarray,
        n_knots: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95
    ) -> dict:
        """
        Build spline configuration from training data.
        
        Args:
            data: array with price data (e.g.: log_price_per_liter)
            n_knots: number of knots to generate
            q_min: minimum quantile for knots
            q_max: maximum quantile for knots
        
        Returns:
            dict with:
                'knots': tensor with positions of knots
                'mean': mean of the data
                'std': standard deviation of the data
        """
        # Calculate statistics: mean and standard deviation.
        stats = self.stats_calc.compute_stats(data)
        
        # Generate knots: positions of the knots.
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