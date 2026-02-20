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
        self.enforce_negative_beta = enforce_negative_beta

        self.head_b     = nn.Linear(hidden_dim, n)
        self.head_beta  = nn.Linear(hidden_dim, n)
        self.head_w     = nn.Linear(hidden_dim, n * K_splines)
        n_cross = n * (n - 1) // 2
        self.head_cross = nn.Linear(hidden_dim, n_cross)

        # Pre-compute projection tensor P: (n_cross, n, n)
        # P[k, i, j] = P[k, j, i] = 1 for the k-th (i,j) upper-triangle pair
        # Using einsum('bk,kij->bij', raw_cross, P) builds A out-of-place → vmap-safe
        idx = torch.triu_indices(n, n, offset=1)   # (2, n_cross)
        P = torch.zeros(n_cross, n, n)
        for k in range(n_cross):
            i, j = idx[0, k].item(), idx[1, k].item()
            P[k, i, j] = 1.0
            P[k, j, i] = 1.0
        self.register_buffer('_P', P)              # (n_cross, n, n), moves with .to(device)

    def run(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        B = h.shape[0]

        b        = self.head_b(h)                                  # (B, n)
        beta_raw = self.head_beta(h)                               # (B, n)
        beta     = -F.softplus(beta_raw) if self.enforce_negative_beta else beta_raw
        w        = self.head_w(h).view(B, self.n, self.K_splines) # (B, n, K)

        raw_cross = self.head_cross(h)                             # (B, n_cross)
        A = torch.einsum('bk,kij->bij', raw_cross, self._P)       # (B, n, n), sym, diag=0

        return {'b': b, 'beta': beta, 'w': w, 'A': A}

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)