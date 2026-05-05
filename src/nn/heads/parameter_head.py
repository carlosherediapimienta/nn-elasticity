import torch
import torch.nn as nn
import torch.nn.functional as F

class DemandParameterHead(nn.Module):
    """
    Generates parameters b(x), beta(x), w(x), beta_cross(x), w_cross(x), u(x)
    from the latent representation h.

    Parameters:
    - b:          intercept per product                              (B, n)
    - beta:       linear own-price coefficient β_{ii}(x)            (B, n)
    - w:          own-price spline weights w_{ii}(x)                (B, n, K)
    - beta_cross: linear cross-price coefficient β_{ij}(x)          (B, n_pairs)
                    scales the term β_{ij}(x) · u_j in g_i
    - w_cross:    cross-price spline weights w_{ij}(x)              (B, n_pairs, K)
                    weights B_j(u_j) in the cross-price contribution to g_i
    - u:          bilinear interaction tensor U^{(ij)}(x)           (B, n_pairs, K, K)
                    u_{p,k,l} weights B_k(u_i) · B_l(u_j)
                    If use_cross=False, all cross tensors are empty (no cross-price interactions).

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
        self.head_b    = nn.Linear(hidden_dim, 1) # b: intercept per product - (B, 1)
        self.head_beta = nn.Linear(hidden_dim, 1) # beta: linear own-price coefficient - (B, 1)
        self.head_w    = nn.Linear(hidden_dim, K_splines) # w: own-price spline weights - (B, n, K)
        if use_cross:
            # beta_cross: linear cross-price coefficient per directed pair (i,j), i != j.
            # Captures the marginal effect of u_j on g_i through a scalar β_{ij}(x).
            # Shape after squeeze: (B, n_pairs)
            self.head_beta_cross = nn.Linear(2 * hidden_dim, 1)
            # w_cross: cross-price spline weights per directed pair (i,j), i != j.
            # Weights the spline basis B_j(u_j) in the cross-price contribution to g_i.
            # Shape: (B, n_pairs, K)
            self.head_w_cross = nn.Linear(2 * hidden_dim, K_splines)
            # head_cross: bilinear interaction weights per directed pair (i,j), i != j.
            # Reshaped to (B, n_pairs, K, K) to weight B_i(u_i)^T U^{(ij)} B_j(u_j).
            self.head_cross = nn.Linear(2 * hidden_dim, K_splines * K_splines)

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

    def run(self, h: torch.Tensor, pairs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # Number of samples in the batch, coming from the latten representation h.
        # Number of products, coming from the latten representation h.
        # Hidden dimension, coming from the latten representation h.
        B, n, H = h.shape 
        K = self.K_splines # Number of splines.

        # pairs selections. Otherwise, we get everything.
        active_pairs = pairs if pairs is not None else self._pairs

        b        = self.head_b(h).squeeze(-1) # b: intercept per product - (B, n)
        beta_raw = self.head_beta(h).squeeze(-1) # beta: linear own-price coefficient - (B, n)
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
        w        = self.head_w(h)  # w: own-price spline weights - (B, n, K)

        if self.use_cross:
            i_idx, j_idx = active_pairs[0], active_pairs[1]  # (n_pairs,)
            n_active = active_pairs.shape[1]
            h_i  = h[:, i_idx, :]               # (B, n_pairs, H)
            h_j  = h[:, j_idx, :]               # (B, n_pairs, H)
            h_ij = torch.cat([h_i, h_j], dim=-1) # (B, n_pairs, 2*H)

            # beta_cross: linear cross-price coefficient β_{ij}(x) - (B, n_pairs)
            # Scales the linear term β_{ij}(x) · u_j in the demand formula.
            beta_cross = self.head_beta_cross(h_ij).squeeze(-1)

            # w_cross: cross-price spline weights w_{ij}(x) - (B, n_pairs, K)
            # Weights the spline basis B_j(u_j) so that the model can capture
            # non-linear cross-price effects beyond the linear β_{ij} term.
            w_cross = self.head_w_cross(h_ij)  # (B, n_pairs, K)

            # u: bilinear interaction tensor U^{(ij)}(x) - (B, n_pairs, K, K)
            # u_{p,k,l} weights B_k(u_i) * B_l(u_j) in the cross potential.
            u = self.head_cross(h_ij).view(B, n_active, K, K)
        else:
            # Empty tensors — no cross-price parameters allocated.
            beta_cross = torch.empty(B, 0, device=h.device, dtype=h.dtype)          # (B, 0)
            w_cross    = torch.empty(B, 0, K, device=h.device, dtype=h.dtype)       # (B, 0, K)
            u          = torch.empty(B, 0, K, K, device=h.device, dtype=h.dtype)    # (B, 0, K, K)

        return {
            'b':          b,
            'beta':       beta,
            'beta_cross': beta_cross,  # linear cross-price coefficients β_{ij}(x) - (B, n_pairs)
            'w':          w,
            'w_cross':    w_cross,     # cross-price spline weights w_{ij}(x)       - (B, n_pairs, K)
            'u':          u,           # bilinear interaction tensor U^{(ij)}(x)    - (B, n_pairs, K, K)
            'pairs':      active_pairs,
        }

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)