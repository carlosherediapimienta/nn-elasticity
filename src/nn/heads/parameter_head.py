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
              If use_cross=False, u is always zero (no cross-price interactions).

    Public API: run().
    """

    def __init__(
        self,
        hidden_dim: int,
        K_splines: int,
        n: int,
        enforce_negative_beta: bool = False,
        use_cross: bool = True,
    ):
        super().__init__()
        self.n = n
        self.K_splines = K_splines
        self.use_cross = use_cross
        # If cross-price terms are disabled, treat n_cross as 0 so we don't
        # allocate or compute cross tensors at all.
        self.n_cross = (n * (n - 1) // 2) if use_cross else 0

        self.head_b    = nn.Linear(hidden_dim, n)
        self.head_beta = nn.Linear(hidden_dim, n)
        self.head_w    = nn.Linear(hidden_dim, n * K_splines)
        if use_cross:
            self.head_cross = nn.Linear(hidden_dim, self.n_cross * K_splines * K_splines)

        self.enforce_negative_beta = enforce_negative_beta

        if use_cross:
            idx = torch.triu_indices(n, n, offset=1)
        else:
            # Empty (2, 0)
            idx = torch.empty(2, 0, dtype=torch.long)
        self.register_buffer('_pairs', idx)

    def run(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        B = h.shape[0]
        K = self.K_splines

        b        = self.head_b(h)
        beta_raw = self.head_beta(h)
        beta     = -F.softplus(beta_raw) if self.enforce_negative_beta else beta_raw
        w        = self.head_w(h).view(B, self.n, K)

        if self.use_cross:
            u = self.head_cross(h).view(B, self.n_cross, K, K)
        else:
            # Empty tensor: downstream code can skip all cross computations.
            u = torch.empty(B, 0, K, K, device=h.device, dtype=h.dtype)

        return {'b': b, 'beta': beta, 'w': w, 'u': u, 'pairs': self._pairs}

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)