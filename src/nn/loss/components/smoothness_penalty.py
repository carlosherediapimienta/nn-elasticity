import torch
from .curvature_calculator import CurvatureCalculator

class SmoothnessPenalty:
    """
    Regularization penalty that discourages highly curved demand curves.

        L_smooth = sum_{b,i} m_{bi} kappa_{bi}^2  /  sum_{b,i} m_{bi}

    where kappa_i = d^2 log q_i / d log p_i^2 (own spline + cross interactions)
    and m is obs_mask: unobserved demand cells do not enter the mean.

    Cross-term contributions to kappa are zeroed for unavailable j inside
    CurvatureCalculator (availability_j). Do not re-average over (i, j) here.
    """

    def __init__(self):
        self.curvature_calc = CurvatureCalculator()

    def run(
        self,
        w: torch.Tensor,      # (B, n, K)
        ddBx: torch.Tensor,   # (B, n, K)
        u: torch.Tensor,      # (B, n_cross, K, K)
        Bx: torch.Tensor,     # (B, n, K)
        pairs: torch.Tensor,  # (2, n_cross)
        attn_weights: torch.Tensor | None = None,  # (B, n_cross)
        availability: torch.Tensor | None = None,  # (B, n)
        obs_mask: torch.Tensor | None = None,      # (B, n)
        price_observed: torch.Tensor | None = None,  # (B, n), optional
    ) -> torch.Tensor:
        kappa = self.curvature_calc.run(
            w, ddBx, u, Bx, pairs, attn_weights, availability,
        )  # (B, n)

        if obs_mask is None:
            return kappa.square().mean()

        m = obs_mask.to(dtype=kappa.dtype)
        if price_observed is not None:
            m = m * price_observed.to(dtype=kappa.dtype)

        return (kappa.square() * m).sum() / m.sum().clamp_min(1.0)