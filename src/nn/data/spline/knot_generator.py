import numpy as np
import torch

class KnotGenerator:
    """
    Generates knots for splines based on quantiles of the data.
    Public API: generate_from_quantiles().
    """
    def generate_from_quantiles(self, data: np.ndarray, n_knots=16, q_min=0.05, q_max=0.95):
        """
        Generates knots for splines based on quantiles of the data.
        Args:
            data: array of price values (log_price) for a single product.
            n_knots: number of knots to generate
            q_min: minimum quantile for knots
            q_max: maximum quantile for knots
        Returns:
            tensor with positions of knots
        """
        x = np.asarray(data, dtype=np.float64) # Convert the data to a numpy array
        x = x[np.isfinite(x)]
        if x.size == 0: # If there are no finite values, raise an error for the user
            raise ValueError("There are no finite values for the spline knots.")

        quantiles = np.linspace(q_min, q_max, n_knots) # Generate the quantiles
        knot_values = np.quantile(x, quantiles) # Compute the quantiles of the data
        
        # Example:
        # x = [1.0, 1.2, 1.5, 1.5, 1.8, 2.0, 2.0, 2.3, 2.5, 3.0]
        # Step 1: generates 5 points evenly spaced between 0.05 and 0.95
        # quantiles = np.linspace(0.05, 0.95, 5)
        # Result: [0.05, 0.275, 0.50, 0.725, 0.95]
        # Step 2: calculates the price values at those percentiles
        # knot_values = np.quantile(x, quantiles)
        # Result: [1.04, 1.49, 1.90, 2.18, 2.77]

        # Important! With these knots, the spline will be able to capture better the place where 
        # the price is really changing. Otherwise, the spline would be too smooth.
        return torch.tensor(knot_values, dtype=torch.float32) # Return the knots as a tensor

    def generate_boundary_and_interior_knots(
        self, 
        data: np.ndarray,
        n_interior_knots: int = 16,
        q_min: float = 0.05,
        q_max: float = 0.95,
    ) -> torch.Tensor:
        """
        Generates the K+2 knots (2 boundary + K interior) required by a
        natural cubic spline basis (MultiNaturalCubicSplineBasis).
        Args:
            data: array of price values (log_price) for a single product,
                already restricted to the training fold.
            n_interior_knots: number of interior knots (K). The returned
                tensor has n_interior_knots + 2 entries.
            q_min: minimum quantile for interior knots
            q_max: maximum quantile for interior knots
        Returns:
            tensor of shape (n_interior_knots + 2,), strictly increasing:
                knots[0]    = min(data)   (left boundary, xi_1)
                knots[1:-1] = interior knots (xi_2 .. xi_{K+1})
                knots[-1]   = max(data)   (right boundary, xi_{K+2})
        Raises:
            ValueError: if data has no finite values, or if min == max
                (degenerate training support, would produce null denominators
                downstream in the natural-spline basis).
        """
        x = np.asarray(data, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("There are no finite values for the spline knots.")
        
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            raise ValueError(
                f"Cannot build a natural cubic spline: degenerate training "
                f"support (min={lo}, max={hi}). Prices are constant."
            )

        quantiles = np.linspace(q_min, q_max, n_interior_knots)
        interior = np.quantile(x, quantiles)

        # Discrete/quantized prices routinely produce duplicated quantiles
        # (e.g. many identical price points). Deduplicate and drop any 
        # interior "knot" collapses onto a boundary.
        interior = np.unique(interior)
        interior = interior[(interior > lo) & (interior < hi)]

        if interior.size < n_interior_knots:
            # Deterministic fallback: evenly spaced knots strictly inside
            # (lo, hi). Always yields n_interior_knots strictly increasing,
            # distinct values, so denominator in the natural-spline basis 
            # are neve null.
            interior = np.linspace(lo, hi, n_interior_knots + 2)[1:-1]
        
        knots = np.concatenate(([lo], interior, [hi]))
        if np.any(np.diff(knots) <= 0):
            raise ValueError("Generated knots are not strictly increasing.")
        
        return torch.tensor(knots, dtype=torch.float32)