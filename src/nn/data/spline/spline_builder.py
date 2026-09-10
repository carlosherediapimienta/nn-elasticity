from typing import Literal
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
        n_basis: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95,
        basis_type: Literal["truncated_cubic", "natural_cubic"] = "truncated_cubic"
    ) -> dict:
        """
        Build spline configuration from training data.
        Args:
            data: array with price data (e.g.: log_price_per_liter), already
                restricted to the training fold.
            n_basis: dimension of the nonlinear basis (K). Named N_BASIS
                (not N_KNOTS) because it must match the *basis dimension*
                across basis_type choices, not the raw knot count:
                    - truncated_cubic: n_basis knots  -> basis of dim n_basis
                    - natural_cubic:   n_basis interior + 2 boundary knots
                                       -> basis of dim n_basis (the 2 boundary
                                       knots do not add extra basis functions)
            q_min: minimum quantile for knots
            q_max: maximum quantile for knots
            basis_type: "truncated_cubic" (current ICDN basis) or
                "natural_cubic" (zero curvature outside boundary knots).
        Returns:
            dict with:
                'knots': tensor with positions of knots.
                    - truncated_cubic: shape (n_basis,)
                    - natural_cubic:   shape (n_basis + 2,)  [xi_1 .. xi_{K+2}]
                'mean': mean of the data (shift)
                'std': standard deviation of the data (scale)
                'basis_type': basis_type used to build this config
                'interior_knots': (natural_cubic only) shape (n_basis,)
                'boundary_knots': (natural_cubic only) shape (2,) [xi_1, xi_{K+2}]
        """
        # Calculate statistics: mean and standard deviation.
        stats = self.stats_calc.compute_stats(data)

        # Selection of basis type:
        if basis_type == "truncated_cubic":
            # Generate knots: positions of the knots.
            knots = self.knot_gen.generate_from_quantiles(
                data, n_knots=n_basis, q_min=q_min, q_max=q_max
            )
            config = {
                "knots": knots,
                "mean": stats["mean"],
                "std": stats["std"],
                "basis_type": basis_type
            }
        elif basis_type == "natural_cubic":
            # Alternative basis type: natural cubic splines with zero curvature
            knots = self.knot_gen.generate_boundary_and_interior_knots(
                data, n_interior_knots=n_basis, q_min=q_min, q_max=q_max
            )
            if torch.any(knots[1:] <= knots[:-1]):
                raise ValueError("Generated knots are not strictly increasing.")
            config = {
                "knots": knots,
                "mean": stats["mean"],
                "std": stats["std"],
                "basis_type": basis_type,
                "boundary_knots": knots[[0,-1]].clone(),
                "interior_knots": knots[1:-1].clone(),
            }
        else:
            raise ValueError(f"Unknown basis type: {basis_type!r}")
        return config