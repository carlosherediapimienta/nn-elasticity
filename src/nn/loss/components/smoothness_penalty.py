import torch
from .curvature_calculator import CurvatureCalculator

class SmoothnessPenalty:
    """
    Regularization penalty that discourages highly curved demand curves.
    Computes the mean squared second derivative of predicted log-demand with
    respect to log-price, averaged over products and batch:
        L_smooth = mean_{b,i} [ (d^2log q_i / d log p_i^2)^2 ]
    Both own-price spline curvature (via w, ddBx, dddBx) and cross-price
    interaction contributions (via u, Bx, IBx) are included in the curvature
    estimate through CurvatureCalculator.
    A higher penalty drives the model toward smoother, more monotone demand
    curves and reduces overfitting in low-data price regions.
    Public API:
        run(w, ddBx, dddBx, u, Bx, IBx, pairs) -> torch.Tensor (scalar)
    """

    def __init__(self):
        # Curvature calculator to compute the curvature of the demand curve.
        self.curvature_calc = CurvatureCalculator()

    def run(
        self,
        w: torch.Tensor,      # (B, n, K)
        ddBx: torch.Tensor,   # (B, n, K)
        u: torch.Tensor,      # (B, n_cross, K, K)
        Bx: torch.Tensor,     # (B, n, K)
        pairs: torch.Tensor,  # (2, n_cross)
    ) -> torch.Tensor:
        # Compute the curvature of the demand curve.
        kappa = self.curvature_calc.run(w, ddBx, u, Bx, pairs)  # (B, n)
        # Return the mean of the squared curvature.
        return (kappa ** 2).mean()