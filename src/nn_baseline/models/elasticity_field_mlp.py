import torch
import torch.nn as nn
from typing import Optional


class ElasticityFieldMLP(nn.Module):
    """
    Learns E_phi(x,c) in R^{n x n}.
    Uses smooth activations (tanh/softplus) so that derivatives exist.
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
            n: dimension of the price space
            d_context: dimension of the context vector (0 = no context)
            hidden: number of hidden neurons per layer
            depth: number of hidden layers
            act: activation ('tanh', 'softplus')
        """
        super().__init__()
        self.n = n
        self.d_context = d_context

        acts = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
        }
        if act.lower() not in acts:
            raise ValueError(f"Activation '{act}' not supported. Use: {list(acts.keys())}")
        activation = acts[act.lower()]

        din = n + d_context
        layers = [nn.Linear(din, hidden), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation]
        layers += [nn.Linear(hidden, n * n)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, n) log-prices
            c: (B, d_context) vector of context, or None

        Returns:
            E: (B, n, n) elasticity field
        """
        if self.d_context > 0:
            if c is None:
                raise ValueError("d_context > 0 but c=None")
            inp = torch.cat([x, c], dim=-1)
        else:
            inp = x
        out = self.net(inp)
        return out.view(x.shape[0], self.n, self.n)