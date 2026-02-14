import torch

class CurvatureCalculator:
    """
    Calculate the second derivative d²y/dx² using spline bases.
    Public API: run().
    """
    
    def run(self, w: torch.Tensor, ddBx: torch.Tensor) -> torch.Tensor:
        """
        Calculate d²y/dx² = sum_k w_k(c) * B''_k(x).
        
        Args:
            w: (B, K) spline weights
            ddBx: (B, K) second derivatives of spline bases
        
        Returns:
            d2y_dx2: (B,) second derivative of demand w.r.t. price
        """
        return (w * ddBx).sum(dim=-1)