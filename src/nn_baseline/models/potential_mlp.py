import torch
import torch.nn as nn
from typing import Optional


class PotentialMLP(nn.Module):
    """
    Potential-first model: y = g(x, c).
    La elasticidad se obtiene como E = J_x g (ver JacobianElasticityCalculator).
    Integrable by construction si g es C^2.
    Public API: forward()
    """

    def __init__(
        self,
        n: int,
        d_context: int = 0,
        hidden: int = 256,
        depth: int = 3,
        act: str = "tanh"
    ):
        """
        Args:
            n: dimensión del espacio de precios / demanda
            d_context: dimensión del vector de contexto (0 = sin contexto)
            hidden: neuronas por capa oculta
            depth: número de capas ocultas
            act: activación ('tanh', 'softplus', 'relu')
        """
        super().__init__()
        self.n = n
        self.d_context = d_context

        acts = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "relu": nn.ReLU(),
        }
        if act.lower() not in acts:
            raise ValueError(f"Activation '{act}' not supported. Use: {list(acts.keys())}")
        activation = acts[act.lower()]

        din = n + d_context
        layers = [nn.Linear(din, hidden), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation]
        layers += [nn.Linear(hidden, n)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, n) log-precios
            c: (B, d_context) vector de contexto, o None

        Returns:
            y: (B, n) demanda predicha en log-space
        """
        inp = torch.cat([x, c], dim=-1) if (self.d_context > 0) else x
        return self.net(inp)