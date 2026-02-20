import torch
import torch.nn as nn
from typing import Optional


class Y0MLP(nn.Module):
    """
    Model for anchoring y0(c) = \hat y(x0, c) in the baseline elasticity-first.

    - If d_context > 0: MLP(c) -> R^n.
    - If d_context == 0: global learnable parameter in R^n (bias).
    """

    def __init__(
        self,
        n: int,
        d_context: int = 0,
        hidden: int = 128,
        depth: int = 2,
        act: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n = n
        self.d_context = d_context

        if d_context == 0:
            self.y0 = nn.Parameter(torch.zeros(n))
            self.net = None
            return

        acts = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "relu": nn.ReLU(),
        }
        if act.lower() not in acts:
            raise ValueError(f"Activation '{act}' not supported. Use: {list(acts.keys())}")
        activation = acts[act.lower()]

        layers = [nn.Linear(d_context, hidden), activation, nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, n)]
        self.net = nn.Sequential(*layers)

    def forward(self, c: Optional[torch.Tensor], batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            c: (B, d_context) or None if d_context == 0
            batch_size: necessary only if d_context == 0 and c is None

        Returns:
            y0: (B, n)
        """
        if self.d_context == 0:
            if c is not None:
                B = c.shape[0]
            else:
                if batch_size is None:
                    raise ValueError("batch_size required if d_context==0 and c=None")
                B = batch_size
            return self.y0.unsqueeze(0).expand(B, -1)

        if c is None:
            raise ValueError("d_context > 0 but c=None")
        return self.net(c)
