import torch
import torch.nn as nn
from typing import Optional, Sequence


class EulerIntegrator:
    """
    Integración numérica coordenada a coordenada (Euler).
    y_{t+1} = y_t + E(x_t, c) Δx_t

    Si E no es integrable, el resultado depende del orden y la discretización.
    Public API: run()
    """

    def __init__(self, steps_per_dim: int = 16):
        """
        Args:
            steps_per_dim: pasos de Euler por cada dimensión de x
        """
        if steps_per_dim < 1:
            raise ValueError("steps_per_dim debe ser >= 1")
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
            E_model: modelo que devuelve E(x, c): (B, n, n)
            x0: (B, n) log-precios origen
            y0: (B, n) demanda en el punto origen
            xT: (B, n) log-precios destino
            c:  (B, d) vector de contexto, o None
            order: permutación de [0..n-1] indicando el orden de coordenadas

        Returns:
            y: (B, n) demanda predicha en xT
        """
        B, n = x0.shape
        if order is None:
            order = list(range(n))
        if sorted(order) != list(range(n)):
            raise ValueError("order debe ser una permutación de [0..n-1]")

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