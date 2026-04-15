import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandParameterHead(nn.Module):
    """
    Generates parameters b(x), beta(x), w(x), u(x) from the latent representation x.

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
        self.n = n # number of products
        self.K_splines = K_splines # number of splines
        self.use_cross = use_cross # whether to use cross-price terms

        # If cross-price terms are disabled, treat n_cross as 0 so we don't
        # allocate or compute cross tensors at all. The formula is:
        # n_cross = n * (n - 1) and the reason is because of combinatorics without repetition.
        # For instance, if n = 3 (A, B, C), then the only possible pairs are:
        # (A, B), (B, A) (A, C), (C, A) (B, C), (C, B). Therefore, n_cross = 6. No symmetry!
        self.n_cross = (n * (n - 1)) if use_cross else 0 # number of cross-price terms

        # We build the heads for the parameters.
        self.head_b    = nn.Linear(hidden_dim, n) # b: intercept per product - (B, n)
        self.head_beta = nn.Linear(hidden_dim, n) # beta: linear own-price coefficient - (B, n)
        self.head_w    = nn.Linear(hidden_dim, n * K_splines) # w: own-price spline weights - (B, n, K)
        if use_cross:
            # alpha: cross-price coefficient per pair (j \neq i) - (B, n_cross)
            self.head_alpha = nn.Linear(hidden_dim, self.n_cross)
            # u: cross-price weight tensor per pair (j \neq i) - (B, n_cross, K, K)
            self.head_cross = nn.Linear(hidden_dim, self.n_cross * K_splines * K_splines) 

        # enforce_negative_beta constrains the linear price coefficient to be negative 
        # (FMCG Theory), which prevents non-physical demand curves during early training (phase 0).
        # Note: beta alone does not determine the elasticity — the spline weights w also
        # contribute to the slope. Constraining beta gives a stable baseline while still
        # allowing the splines to capture non-monotone or positive local effects.

        self.enforce_negative_beta = enforce_negative_beta

        # We build the index pairs used for cross-price terms.
        #
        # For an n x n grid of pair indices, for example with n = 3:
        #
        #        j=0    1      2
        # i=0   [0,0] [0,1] [0,2]
        # i=1   [1,0] [1,1] [1,2]
        # i=2   [2,0] [2,1] [2,2]
        #
        # Here we keep all off-diagonal pairs, i.e. all (i, j) such that i != j.
        # This includes both directions of each pair:
        #   (0,1) and (1,0)
        #   (0,2) and (2,0)
        #   (1,2) and (2,1)
        #
        # We first build a boolean mask where the diagonal is False and every
        # off-diagonal position is True:
        #   ~torch.eye(n, dtype=torch.bool)
        #
        # Then mask.nonzero(...).T returns the coordinates of all True entries
        # with shape (2, n * (n - 1)), where:
        #   - idx[0] contains the row indices i
        #   - idx[1] contains the column indices j
        #
        # For n = 3, the selected pairs are:
        #   (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        #
        # So unlike the previous version, we exclude only self-pairs (i == j),
        # but we keep both the upper and lower triangles.
        if use_cross:
            mask = ~torch.eye(n, dtype=torch.bool)
            idx = mask.nonzero(as_tuple=False).T
        else:
            # Empty tensor of shape (2, 0)
            idx = torch.empty(2, 0, dtype=torch.long)
        # We register the pairs as a buffer.
        self.register_buffer('_pairs', idx)

    def run(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        B = h.shape[0] # Number of samples in the batch, coming from the latten representation h.
        K = self.K_splines # Number of splines.

        b        = self.head_b(h) # b: intercept per product - (B, n)
        beta_raw = self.head_beta(h) # beta: linear own-price coefficient - (B, n)
        # -------------------------- IMPORTANT --------------------------
        # softplus(x) = log(1 + exp(x)) is always positive, so -softplus(beta_raw)
        # is always negative - this enforces the prior that own-price coefficients
        # reduce demand.
        #
        # Magnitude behavior:
        #   beta_raw >> 0  →  -softplus(beta_raw) \aprox -beta_raw  (large negative)
        #   beta_raw << 0  →  -softplus(beta_raw) \aprox  0         (near zero, demand insensitive to price)
        #
        # Note: beta is not the elasticity - it is only the linear component of the
        # price effect. The spline weights w also contribute to the slope, so even
        # when beta approximately 0 the model can still capture price sensitivity via the splines.
        beta     = -F.softplus(beta_raw) if self.enforce_negative_beta else beta_raw
        w        = self.head_w(h).view(B, self.n, K) # w: own-price spline weights - (B, n, K)

        if self.use_cross:
            alpha = self.head_alpha(h) # alpha: cross-price coefficient per pair (i!=j) - (B, n_cross)
            # u: cross-price weight tensor per pair (i!=j) - (B, n_cross, K, K)
            u = self.head_cross(h).view(B, self.n_cross, K, K) 
        else:
            # Empty tensor of shape (B, 0).
            alpha = torch.empty(B, 0, device=h.device, dtype=h.dtype)
            # Empty tensor of shape (B, 0, K, K).
            u = torch.empty(B, 0, K, K, device=h.device, dtype=h.dtype)

        return {'b': b, 'beta': beta, 'alpha': alpha, 'w': w, 'u': u, 'pairs': self._pairs}

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)