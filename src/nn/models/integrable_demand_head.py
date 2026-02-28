import torch
import torch.nn as nn
from ..context import ContextMLP
from ..heads import DemandParameterHead, DemandCalculator


class IntegrableDemandHead(nn.Module):
    """
    Multiproduct demand head derived from an integrable scalar potential.

    Potential:
      Φ(x,c) = Σ_i [b_i·x_i + beta_i/2·x_i² + Σ_k w_{ik}·Ψ_k(x_i)]
             + Σ_{p=(i<j)} Σ_{k,l} u_{p,k,l} · Ψ_k(x_i) · B_l(x_j)

    Demand  y = ∂Φ/∂x  →  exact Slutsky symmetry by construction.

    Own-price elasticity:
      ∂y_i/∂x_i = beta_i(c) + Σ_k w_{ik}·B'_k(x_i)
                + Σ_{j: p=(i,j)} Σ_{k,l} u_{p,k,l}·B'_k(x_i)·B_l(x_j)
                + Σ_{j: p=(j,i)} Σ_{k,l} u_{p,k,l}·Ψ_k(x_j)·B''_l(x_i)

    Cross-price elasticity (symmetric):
      ∂y_i/∂x_j = Σ_{k,l} u_{p,k,l}·B_k(x_i)·B'_l(x_j)   for p=(i<j)

    Delegation:
    - ContextMLP:           encodes context c → h
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
        context_encoder: ContextMLP | None = None,
        parameter_head: DemandParameterHead | None = None,
        demand_calc: DemandCalculator | None = None,
    ):
        super().__init__()

        self.ctx = context_encoder or ContextMLP(
            context_dim, hidden=hidden, act=act, dropout=dropout
        )
        H = self.ctx.out_dim

        self.param_head = parameter_head or DemandParameterHead(
            hidden_dim=H,
            K_splines=K_splines,
            n=n,
            enforce_negative_beta=enforce_negative_beta,
            use_cross=use_cross,
        )

        self.demand_calc    = demand_calc    or DemandCalculator()

    def run(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        dBx: torch.Tensor,
        ddBx: torch.Tensor,
        IBx: torch.Tensor,
        return_E: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:

        h      = self.ctx(c)
        params = self.param_head.run(h)   # {b, beta, w, u, pairs}

        y_hat, eps_hat, E = self.demand_calc.run(
            b=params['b'],
            beta=params['beta'],
            w=params['w'],
            x=x,
            Bx=Bx,
            dBx=dBx,
            ddBx=ddBx,
            IBx=IBx,
            u=params['u'],
            pairs=params['pairs'],
            return_E=return_E,
        )
        if return_E and (E is not None):
            params['E'] = E
        return y_hat, eps_hat, params

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)