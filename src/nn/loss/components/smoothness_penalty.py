import torch

class SmoothnessPenalty:
    """
    Smoothness penalty: mean over products and batch of (d²y_i/dx_i²)².
    Includes own-price and cross-price interaction contributions.
    Public API: run().
    """

    def __init__(self, curvature_calc=None):
        from .curvature_calculator import CurvatureCalculator
        self.curvature_calc = curvature_calc or CurvatureCalculator()

    def run(
        self,
        w: torch.Tensor,      # (B, n, K)
        ddBx: torch.Tensor,   # (B, n, K)
        dddBx: torch.Tensor,  # (B, n, K)
        u: torch.Tensor,      # (B, n_cross, K, K)
        Bx: torch.Tensor,     # (B, n, K)
        IBx: torch.Tensor,    # (B, n, K)
        pairs: torch.Tensor,  # (2, n_cross)
    ) -> torch.Tensor:
        kappa = self.curvature_calc.run(w, ddBx, dddBx, u, Bx, IBx, pairs)  # (B, n)
        return (kappa ** 2).mean()