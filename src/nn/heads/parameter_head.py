import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandParameterHead(nn.Module):
    """
    Generates parameters b(c), beta(c), w(c), u(c) from the context representation.

    Parameters:
    - b:    intercept per product                          (B, n)
    - beta: linear own-price coefficient                   (B, n)
    - w:    own-price spline weights                       (B, n, K)
    - u:    cross-price weight tensor per pair (i<j)       (B, n_cross, K, K)
              u_{p,k,l} weights B_k(x_i) * B_l(x_j) in the cross potential

    Public API: run().
    """

    def __init__(
        self,
        hidden_dim: int,
        K_splines: int,
        n: int,
        enforce_negative_beta: bool = False,
    ):
        super().__init__()
        self.n = n
        self.K_splines = K_splines
        self.n_cross = n * (n - 1) // 2

        self.head_b     = nn.Linear(hidden_dim, n)
        self.head_beta  = nn.Linear(hidden_dim, n)
        self.head_w     = nn.Linear(hidden_dim, n * K_splines)
        self.head_cross = nn.Linear(hidden_dim, self.n_cross * K_splines * K_splines)

        self.enforce_negative_beta = enforce_negative_beta

        # Pre-compute pair indices i<j
        idx = torch.triu_indices(n, n, offset=1)  # (2, n_cross)
        self.register_buffer('_pairs', idx)

    def run(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        B = h.shape[0]
        K = self.K_splines

        b        = self.head_b(h)                                  # (B, n)
        beta_raw = self.head_beta(h)                               # (B, n)
        beta     = -F.softplus(beta_raw) if self.enforce_negative_beta else beta_raw
        w        = self.head_w(h).view(B, self.n, K) # (B, n, K)
        u        = self.head_cross(h).view(B, self.n_cross, K, K) # (B, n_cross, K, K)

        return {'b': b, 'beta': beta, 'w': w, 'u': u, 'pairs': self._pairs}

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)