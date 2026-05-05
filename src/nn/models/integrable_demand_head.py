import torch
import torch.nn as nn
from ..context import SharedProductEncoder
from ..heads import DemandParameterHead, DemandCalculator, SparseNeighborSelector


class IntegrableDemandHead(nn.Module):
    """
    Multiproduct demand head derived from an integrable scalar potential.

    Delegation:
    - SharedProductEncoder: encodes context c to h
    - DemandParameterHead:  produces b, beta, w, u from h
    - DemandCalculator:     computes y_hat, eps_hat, E

    Public API: run().
    """

    def __init__(
        self,
        context_dim: int,
        K_splines: int,
        n: int,
        k_neighbors: int = 2,
        hidden=(256, 128, 64),
        act="tanh",
        dropout=0.0,
        enforce_negative_beta: bool = False,
        use_cross: bool = True,
    ):
        super().__init__()

        # Build the context MLP.
        self.encoder = SharedProductEncoder(
            context_dim, hidden=hidden, act=act, dropout=dropout
        )
        H = self.encoder.out_dim # Hidden dimension of the shared product encoder.

        # Build the demand parameter head for the parameters b, beta, w, u.
        # It will output a tensor of shape (B, n) for b, (B, n) for beta,
        # (B, n, K) for w, and (B, n_cross, K, K) for u.
        self.param_head = DemandParameterHead(
            hidden_dim=H,
            K_splines=K_splines,
            n=n,
            enforce_negative_beta=enforce_negative_beta,
            use_cross=use_cross,
        )

        # Build the demand calculator to compute the predicted demand y_hat,
        # the own-price elasticity eps_hat, and the elasticity matrix E.
        self.demand_calc    = DemandCalculator()

        # Build the sparse neighbor selector to select the neighbors.
        if use_cross:
            self.neighbor_selector = SparseNeighborSelector(d_hidden=H, k_neighbors=k_neighbors)
        else:
            self.neighbor_selector = None

    def run(
        self,
        tokens: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        dBx: torch.Tensor,
        return_E: bool = False,
        neighbor_meta: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:

        # ── Step 1: Compute the latent representation h from the context c.
        # The context c is a tensor of shape (B, n, d_token).
        # The latent representation h is a tensor of shape (B, n, H).
        # We use the shared product encoder to compute the latent representation h.
        h      = self.encoder(tokens)

        # ── Step 2: Select the neighbors.
        # The neighbors are a tensor of shape (2, n*k_neighbors).
        # We use the sparse neighbor selector to select the neighbors.
        pairs, attn_weights = None, None
        if self.neighbor_selector is not None and neighbor_meta is not None:
            pairs, attn_weights = self.neighbor_selector.run(
                h=h,
                category=neighbor_meta["category"],
                brand=neighbor_meta["brand"],
                style=neighbor_meta["style"],
                liters=neighbor_meta["liters"],
            )

        # ── Step 2: Compute the parameters b, beta, w, u from the latent representation h.
        # The parameters b, beta, w, u are a tensor of shape (B, n) for b,
        # (B, n) for beta, (B, n, K) for w, and (B, n_cross, K, K) for u.
        # We use the demand parameter head to compute the parameters b, beta, w, u.
        params = self.param_head.run(h, pairs=pairs)   # {b, beta, w, u, pairs}

        # ── Step 3: Compute the predicted demand y_hat, the own-price elasticity eps_hat,
        # and the elasticity matrix E.
        # The predicted demand y_hat is a tensor of shape (B, n).
        # The own-price elasticity eps_hat is a tensor of shape (B, n).
        # The elasticity matrix E is a tensor of shape (B, n, n).
        # We use the demand calculator to compute the predicted demand y_hat,
        # the own-price elasticity eps_hat, and the elasticity matrix E.
        y_hat, eps_hat, E = self.demand_calc.run(
            b=params['b'],
            beta=params['beta'],
            w=params['w'],
            x=x,
            Bx=Bx,
            dBx=dBx,
            beta_cross=params['beta_cross'],  # linear cross-price coefficient β_{ij}(x) - (B, n_pairs)
            w_cross=params['w_cross'],         # cross-price spline weights w_{ij}(x)     - (B, n_pairs, K)
            u=params['u'],
            pairs=params['pairs'],
            attn_weights=attn_weights,
            return_E=return_E,
        )
        if return_E and (E is not None):
            params['E'] = E
        return y_hat, eps_hat, params

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)