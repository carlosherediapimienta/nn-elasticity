import torch
import torch.nn as nn
from typing import Optional, Sequence


class EulerIntegrator:
    """
    Numerical integration coordinate by coordinate (Euler).
    y_{t+1} = y_t + E(x_t, c) Δx_t

    If E is not integrable, the result depends on the order and the discretization.
    Public API: run()
    """

    def __init__(self, steps_per_dim: int = 16):
        """
        Args:
            steps_per_dim: steps of Euler by each dimension of x
        """
        if steps_per_dim < 1:
            raise ValueError("steps_per_dim must be >= 1")
        self.steps_per_dim = steps_per_dim

    def run(
        self,
        E_model: nn.Module,
        x0: torch.Tensor,
        y0: torch.Tensor,
        xT: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        order: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            E_model: module that returns E(x, c): (B, n, n)
            x0: (B, n) log-prices origin
            y0: (B, n) demand at the origin point
            xT: (B, n) log-prices destination
            c:  (B, d) vector of context, or None
            order: permutation of [0..n-1] indicating the order of coordinates

        Returns:
            y: (B, n) predicted demand at xT
        """
        B, n = x0.shape
        if order is None:
            order = list(range(n))
        if sorted(order) != list(range(n)):
            raise ValueError("order must be a permutation of [0..n-1]")

        x = x0.clone()
        y = y0.clone()

        for j in order:
            total = xT[:, j] - x0[:, j]
            step = total / float(self.steps_per_dim)
            for _ in range(self.steps_per_dim):
                dx = torch.zeros_like(x)
                dx[:, j] = step          
                E = E_model(x, c)
                y = y + torch.einsum("bij,bj->bi", E, dx)
                x = x + dx              
        return y