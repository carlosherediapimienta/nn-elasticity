import torch
import torch.nn as nn
from typing import Optional
from ..integration.euler_integrator import EulerIntegrator


class PathDependenceDiagnostics:
    """
    Mide la dependencia del orden de integración.
    gap = ||y_hat(orden natural) - y_hat(orden inverso)||

    Delegation:
    - EulerIntegrator: integración numérica coordenada a coordenada

    Public API: run()
    """

    def __init__(
        self,
        steps_per_dim: int = 16,
        integrator: EulerIntegrator | None = None
    ):
        """
        Args:
            steps_per_dim: pasos de Euler por dimensión
            integrator: instancia de EulerIntegrator (opcional, dependency injection)
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
            E_model: modelo que devuelve E(x, c): (B, n, n)
            x0: (B, n) log-precios origen
            y0: (B, n) demanda en el punto origen
            xT: (B, n) log-precios destino
            c:  (B, d) vector de contexto, o None

        Returns:
            gap: escalar, media sobre el batch de ||y_a - y_b||
        """
        n = x0.shape[1]
        order_a = list(range(n))
        order_b = list(reversed(range(n)))

        ya = self.integrator.run(E_model, x0, y0, xT, c=c, order=order_a)
        yb = self.integrator.run(E_model, x0, y0, xT, c=c, order=order_b)
        return torch.linalg.norm(ya - yb, dim=-1).mean()