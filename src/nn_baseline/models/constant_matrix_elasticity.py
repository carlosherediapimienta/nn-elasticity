import torch
import torch.nn as nn
from typing import Optional


class ConstantMatrixElasticity(nn.Module):
    """
    E(x,c) = W (constante, full matrix con cross-effects).
    Integrable: y(x) = y0 + W(x - x0).
    Public API: forward()
    """

    def __init__(self, n: int, init_scale: float = 1e-2):
        """
        Args:
            n: número de SKUs / dimensión del espacio de precios
            init_scale: escala de inicialización aleatoria de W
        """
        super().__init__()
        self.n = n
        W = init_scale * torch.randn(n, n)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, n) log-precios
            c: ignorado (API uniforme con otros modelos)

        Returns:
            E: (B, n, n) campo de elasticidad constante
        """
        B = x.shape[0]
        return self.W.unsqueeze(0).expand(B, self.n, self.n)