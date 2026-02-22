import torch
import torch.nn as nn
import torch.nn.functional as F


class DemandParameterHead(nn.Module):
    """
    Generate parameters b(c), beta(c), w(c), A(c) from context representation.

    Parameters:
    - b:    intercept term                           (B, n)
    - beta: linear own-price coefficient             (B, n)
    - w:    spline weights                           (B, n, K)
    - A:    symmetric cross-price matrix, zero diag  (B, n, n)
              A_{ij}(c) = ∂y_i/∂x_j  for i≠j

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