import torch
import torch.nn as nn


class ContextMLP(nn.Module):
    """
    Smooth encoder MLP to transform context.
    Public API: forward() (following nn.Module convention).
    """
    
    def __init__(self, d_in: int, hidden=(256, 128, 64), act="tanh", dropout=0.0):
        """
        Args:
            d_in: dimension of input
            hidden: tuple with dimensions of hidden layers
            act: activation function ('tanh', 'softplus', 'gelu')
            dropout: dropout rate between layers (0.0 = no dropout)
        """
        super().__init__()
        
        acts = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "gelu": nn.GELU(),
        }
        
        if act not in acts:
            raise ValueError(f"Activation '{act}' not supported. Use: {list(acts.keys())}")
        
        a = acts[act]

        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(a)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Process context through the MLP.
        
        Args:
            c: (B, d_in) context tensor
        
        Returns:
            (B, out_dim) processed representation
        """
        return self.net(c)