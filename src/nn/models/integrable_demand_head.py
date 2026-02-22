import torch
import torch.nn as nn
from ..context import ContextMLP
from ..heads import DemandParameterHead, DemandCalculator


class IntegrableDemandHead(nn.Module):
    """
    Potential model head for multiproduct demand with symmetric cross-price effects:

      y_i = b_i(c) + beta_i(c)*x_i + Σ_k w_{ik}(c)*B_k(x_i) + Σ_{j≠i} A_{ij}(c)*x_j

    Derives from scalar potential Φ(x,c) → Slutsky symmetry exact by construction:
      ∂y_i/∂x_j = A_{ij}(c) = A_{ji}(c) = ∂y_j/∂x_i

    Own-price elasticity:
      eps_i = ∂y_i/∂x_i = beta_i(c) + Σ_k w_{ik}(c)*B'_k(x_i)

    Cross-price elasticity matrix (context-dependent, symmetric):
      A(c)  (available in aux['A'])

    Delegation:
    - ContextMLP:          process context
    - DemandParameterHead: generate b, beta, w, A
    - DemandCalculator:    calculate y_hat (with cross terms)
    - ElasticityCalculator: calculate own-price eps_hat

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
        )
        params['E'] = E
        return y_hat, eps_hat, params

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)