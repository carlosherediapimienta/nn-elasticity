import torch
import torch.nn as nn
from typing import Optional, Sequence

from ..integration.euler_integrator import EulerIntegrator


class ElasticityFirstPredictor(nn.Module):
    """Elasticity-first predictor: y_hat(x,c) = y_hat(x0,c) + integral_{x0->x} E_phi(s,c) ds

Implemented with coordinate-wise Euler integration with fixed update order.
"""

    def __init__(
        self,
        E_model: nn.Module,
        y0_model: nn.Module,
        x0: Optional[torch.Tensor],
        integrator: Optional[EulerIntegrator] = None,
        steps_per_dim: int = 16,
        default_order: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.E_model = E_model
        self.y0_model = y0_model
        self.integrator = integrator or EulerIntegrator(steps_per_dim=steps_per_dim)
        self.default_order = None if default_order is None else list(default_order)

        if x0 is not None:
            # buffer para que viaje con .to(device)
            self.register_buffer("x0", x0.clone().detach())
        else:
            self.x0 = None

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        order: Optional[Sequence[int]] = None,
        return_E_at_x: bool = False,
    ):
        """
        Args:
            x: (B, n) log-precios objetivo
            c: (B, d) contexto o None
            x0: (n,) o (B, n) log-precios de referencia. Si None, usa self.x0.
            order: orden de integración (perm. de [0..n-1]). Si None, usa default_order o natural.
            return_E_at_x: si True, devuelve también E(x,c) evaluado en el endpoint.

        Returns:
            y_hat: (B, n) o (y_hat, E_x)
        """
        B, n = x.shape

        # resolver x0
        if x0 is None:
            if self.x0 is None:
                raise ValueError("x0 no provisto y self.x0 es None. Pasa x0 o fija x0 en el init().")
            x0_use = self.x0
        else:
            x0_use = x0

        if x0_use.ndim == 1:
            if x0_use.shape[0] != n:
                raise ValueError(f"x0 (vector) debe tener shape ({n},)")
            x0_batch = x0_use.unsqueeze(0).expand(B, -1)
        elif x0_use.ndim == 2:
            if x0_use.shape != (B, n):
                raise ValueError(f"x0 (batch) debe tener shape {(B, n)}")
            x0_batch = x0_use
        else:
            raise ValueError("x0 debe ser (n,) o (B,n)")

        # y0(c)
        y0 = self.y0_model(c, batch_size=B) if hasattr(self.y0_model, "forward") else self.y0_model(c)

        # orden
        if order is None:
            order_use = self.default_order
        else:
            order_use = list(order)

        y_hat = self.integrator.run(
            E_model=self.E_model,
            x0=x0_batch,
            y0=y0,
            xT=x,
            c=c,
            order=order_use,
        )

        if return_E_at_x:
            E_x = self.E_model(x, c)
            return y_hat, E_x

        return y_hat
