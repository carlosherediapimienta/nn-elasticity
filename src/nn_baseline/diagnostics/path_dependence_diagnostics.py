import torch
import torch.nn as nn
from typing import Optional
from ..integration.euler_integrator import EulerIntegrator


class PathDependenceDiagnostics:
    """
    Measures the order dependence of integration.
    gap = ||y_hat(natural order) - y_hat(reverse order)||

    Delegation:
    - EulerIntegrator: numerical integration coordinate by coordinate

    Public API: run()
    """

    def __init__(
        self,
        steps_per_dim: int = 16,
        integrator: EulerIntegrator | None = None
    ):
        """
        Args:
            steps_per_dim: steps of Euler by dimension
            integrator: instance of EulerIntegrator (optional, dependency injection)
        """
        self.integrator = integrator or EulerIntegrator(steps_per_dim=steps_per_dim)

    def run(
        self,
        E_model: nn.Module,
        x0: torch.Tensor,
        y0: torch.Tensor,
        xT: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            E_model: module that returns E(x, c): (B, n, n)
            x0: (B, n) log-prices origin
            y0: (B, n) demand at the origin point
            xT: (B, n) log-prices destination
            c:  (B, d) vector of context, or None

        Returns:
            gap: scalar, mean over the batch of ||y_a - y_b||
        """
        n = x0.shape[1]
        order_a = list(range(n))
        order_b = list(reversed(range(n)))

        ya = self.integrator.run(E_model, x0, y0, xT, c=c, order=order_a)
        yb = self.integrator.run(E_model, x0, y0, xT, c=c, order=order_b)
        return torch.linalg.norm(ya - yb, dim=-1).mean()