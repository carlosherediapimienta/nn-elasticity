import torch
import torch.nn as nn
from ..context import ContextMLP
from ..heads import DemandParameterHead, DemandCalculator


class IntegrableDemandHead(nn.Module):
    """
    Multiproduct demand head derived from an integrable scalar potential.

    Delegation:
    - ContextMLP:           encodes context c to h
    - DemandParameterHead:  produces b, beta, w, u from h
    - DemandCalculator:     computes y_hat, eps_hat, E

    Public API: run().
    """

    def __init__(
        self,
        context_dim: int,
        K_splines: int,
        n: int,
        hidden=(256, 128, 64),
        act="tanh",
        dropout=0.0,
        enforce_negative_beta: bool = False,
        use_cross: bool = True,
    ):
        super().__init__()

        # Build the context MLP.
        self.ctx = ContextMLP(
            context_dim, hidden=hidden, act=act, dropout=dropout
        )
        H = self.ctx.out_dim # Hidden dimension of the context MLP.

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

    def run(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        dBx: torch.Tensor,
        return_E: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:

        # ── Step 1: Compute the latent representation h from the context c.
        # The context c is a tensor of shape (B, context_dim).
        # The latent representation h is a tensor of shape (B, hidden_dim).
        # We use the context MLP to compute the latent representation h.
        h      = self.ctx(c)

        # ── Step 2: Compute the parameters b, beta, w, u from the latent representation h.
        # The parameters b, beta, w, u are a tensor of shape (B, n) for b,
        # (B, n) for beta, (B, n, K) for w, and (B, n_cross, K, K) for u.
        # We use the demand parameter head to compute the parameters b, beta, w, u.
        params = self.param_head.run(h)   # {b, beta, w, u, pairs}

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
            alpha=params['alpha'],
            u=params['u'],
            pairs=params['pairs'],
            return_E=return_E,
        )
        if return_E and (E is not None):
            params['E'] = E
        return y_hat, eps_hat, params

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)