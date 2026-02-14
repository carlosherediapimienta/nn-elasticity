import torch


class ElasticityCalculator:
    """
    Calculate price elasticity of demand:
      eps_hat = dy_hat/dx = beta(c) + sum_k w_k(c) * B'_k(x)
    
    where x = log_price_per_liter
    
    Public API: run().
    """
    
    def run(
        self,
        beta: torch.Tensor,
        w: torch.Tensor,
        dBx: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate price elasticity of demand.
        
        Args:
            beta: (B,) linear coefficient
            w: (B, K) spline weights
            dBx: (B, K) derivatives of spline bases evaluated at x
        
        Returns:
            eps_hat: (B,) predicted price elasticity
        """
        eps_hat = beta + (w * dBx).sum(dim=-1)
        return eps_hat