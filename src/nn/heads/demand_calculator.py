import torch


class DemandCalculator:
    """
    Calculate predicted demand (log-space) using the potential model:
      y_hat = b(c) + beta(c)*x + sum_k w_k(c) * B_k(x)
    
    Public API: run().
    """
    
    def run(
        self,
        b: torch.Tensor,
        beta: torch.Tensor,
        w: torch.Tensor,
        x: torch.Tensor,
        Bx: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate predicted demand.
        
        Args:
            b: (B,) intercept term
            beta: (B,) linear coefficient
            w: (B, K) spline weights
            x: (B,) or (B,1) log_price_per_liter
            Bx: (B, K) evaluated spline bases at x
        
        Returns:
            y_hat: (B,) predicted demand (log-space)
        """
        return  b + beta * x.float() + (w * Bx).sum(dim=-1)