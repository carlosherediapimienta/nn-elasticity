import torch
from typing import Optional


class DemandCalculator:
    """
    Calculate predicted demand (log-space) using the extended potential model:

      y_hat_i = b_i(c) + beta_i(c)*x_i + Σ_k w_{ik}(c)*B_k(x_i) + Σ_{j≠i} A_{ij}(c)*x_j

    Derives from the scalar potential:
      Φ(x,c) = Σ_i [b_i*x_i + beta_i/2*x_i² + Σ_k w_{ik}*∫B_k dx_i] + (1/2)*x^T A x

    so y = ∂Φ/∂x, guaranteeing ∂y_i/∂x_j = ∂y_j/∂x_i by Schwarz's theorem.

    Public API: run().
    """

    def run(
        self,
        b: torch.Tensor,
        beta: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            b:   (B, n) intercept
            beta:(B, n) own-price linear coefficient
            w:   (B, n, K) spline weights
            x:   (B, n) log_price per product
            Bx:  (B, n, K) evaluated spline bases at x
            A:   (B, n, n) symmetric cross-price matrix, zero diagonal (optional)

        Returns:
            y_hat: (B, n) predicted demand (log-space)
        """
        x = x.float()
        y_hat = b + beta * x + (w * Bx).sum(dim=-1)   # (B, n)

        if A is not None:
            # cross_i = Σ_j A_{ij} * x_j  (A diagonal = 0, so only j≠i contributes)
            cross_term = (A @ x.unsqueeze(-1)).squeeze(-1)  # (B, n)
            y_hat = y_hat + cross_term

        return y_hat