import torch
import torch.nn as nn
import torch.nn.functional as F


class DemandParameterHead(nn.Module):
    """
    Generate parameters b(c), beta(c), w(c) from context representation.
    
    Parameters:
    - b: intercept term
    - beta: linear price coefficient
    - w: spline weights
    
    API pública: run().
    """
    
    def __init__(self, hidden_dim: int, K_splines: int, n: int, enforce_negative_beta=False):
        super().__init__()
        self.n = n
        self.K_splines = K_splines
        self.head_b    = nn.Linear(hidden_dim, n)
        self.head_beta = nn.Linear(hidden_dim, n)
        self.head_w    = nn.Linear(hidden_dim, n * K_splines)
        self.enforce_negative_beta = enforce_negative_beta

    def run(self, h):
        b        = self.head_b(h)                            # (B, n)
        beta_raw = self.head_beta(h)                         # (B, n)
        beta     = -F.softplus(beta_raw) if self.enforce_negative_beta else beta_raw
        w        = self.head_w(h).view(-1, self.n, self.K_splines)  # (B, n, K)
        return {'b': b, 'beta': beta, 'w': w}
        
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)