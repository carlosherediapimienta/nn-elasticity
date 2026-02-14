import torch
import torch.nn as nn
from ..context import ContextMLP
from ..heads import DemandParameterHead, DemandCalculator, ElasticityCalculator


class IntegrableDemandHead1D(nn.Module):
    """
    Potential model head for log-linear demand:
      y_hat = b(c) + beta(c)*x + sum_k w_k(c) * B_k(x)
    
    Price elasticity w.r.t. x=log_price:
      eps_hat = dy_hat/dx = beta(c) + sum_k w_k(c) * B'_k(x)
    
    Delegation:
    - ContextMLP: process context
    - DemandParameterHead: generate parameters b, beta, w
    - DemandCalculator: calculate y_hat
    - ElasticityCalculator: calculate eps_hat
    
    Public API: run().
    """
    
    def __init__(
        self,
        context_dim: int,
        K_splines: int,
        hidden=(256, 128, 64),
        act="tanh",
        dropout=0.0,
        enforce_negative_beta: bool = False,
        # Optional: dependency injection
        context_encoder: ContextMLP | None = None,
        parameter_head: DemandParameterHead | None = None,
        demand_calc: DemandCalculator | None = None,
        elasticity_calc: ElasticityCalculator | None = None
    ):
        """
        Args:
            context_dim: dimension of the context vector
            K_splines: number of spline bases
            hidden: architecture of the context MLP
            act: activation function
            dropout: dropout rate
            enforce_negative_beta: force beta <= 0
            context_encoder: custom encoder (optional)
            parameter_head: custom parameter head (optional)
            demand_calc: custom demand calculator (optional)
            elasticity_calc: custom elasticity calculator (optional)
        """
        super().__init__()
        
        # Dependency injection or default creation
        self.ctx = context_encoder or ContextMLP(
            context_dim,
            hidden=hidden,
            act=act,
            dropout=dropout
        )
        
        H = self.ctx.out_dim
        
        self.param_head = parameter_head or DemandParameterHead(
            hidden_dim=H,
            K_splines=K_splines,
            enforce_negative_beta=enforce_negative_beta
        )
        
        self.demand_calc = demand_calc or DemandCalculator()
        self.elasticity_calc = elasticity_calc or ElasticityCalculator()

    def run(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor,
        dBx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        Public API. Calculate demand and elasticity.
        
        Args:
            c: (B, context_dim) context vector
            x: (B,) o (B,1) log_price_per_liter
            Bx: (B, K) evaluated spline bases at x
            dBx: (B, K) derivatives of bases evaluated at x
        
        Returns:
            y_hat: (B,) predicted demand (log-space)
            eps_hat: (B,) predicted price elasticity
            aux: dict with intermediate parameters {'b', 'beta', 'w'}
        """
        # 1. Process context
        h = self.ctx(c)  # (B, H)
        
        # 2. Generate parameters
        params = self.param_head.run(h)  # {'b', 'beta', 'w'}
        
        # 3. Calculate demand
        y_hat = self.demand_calc.run(
            b=params['b'],
            beta=params['beta'],
            w=params['w'],
            x=x,
            Bx=Bx
        )
        
        # 4. Calculate elasticity
        eps_hat = self.elasticity_calc.run(
            beta=params['beta'],
            w=params['w'],
            dBx=dBx
        )
        
        return y_hat, eps_hat, params
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module's forward method."""
        return self.run(*args, **kwargs)