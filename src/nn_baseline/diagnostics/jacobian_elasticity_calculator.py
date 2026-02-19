import torch
import torch.nn as nn
from typing import Optional
from torch.func import vmap, jacrev


class JacobianElasticityCalculator:
    """
    Calcula E(x,c) = J_x g(x,c) desde un modelo potencial (PotentialMLP).
    E es integrable by construction.
    Requiere PyTorch >= 2.0 (torch.func).
    Public API: run()
    """

    def run(
        self,
        g_model: nn.Module,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            g_model: modelo potencial que devuelve y: (B, n)
            x: (B, n) log-precios
            c: (B, d) vector de contexto, o None

        Returns:
            E: (B, n, n) campo de elasticidad = jacobiano de g respecto a x
        """
        def g_single(x1: torch.Tensor, c1: Optional[torch.Tensor]) -> torch.Tensor:
            xs = x1.unsqueeze(0)
            cs = None if c1 is None else c1.unsqueeze(0)
            return g_model(xs, cs)[0]

        if c is None:
            J = vmap(jacrev(lambda x1: g_single(x1, None)))(
                x.detach().requires_grad_(True)
            )
        else:
            J = vmap(jacrev(lambda x1, c1: g_single(x1, c1), argnums=0))(
                x.detach().requires_grad_(True), c.detach()
            )
        return J