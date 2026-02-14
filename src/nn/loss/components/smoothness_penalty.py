import torch

class SmoothnessPenalty:
    """
    Smoothness penalty: mean((d²y/dx²)²).
    Promotes smoothness in the demand curve.
    Public API: run().
    """
    
    def __init__(self, curvature_calc=None):
        """
        Args:
            curvature_calc: CurvatureCalculator (optional)
        """
        from .curvature_calculator import CurvatureCalculator
        self.curvature_calc = curvature_calc or CurvatureCalculator()
    
    def run(self, w: torch.Tensor, ddBx: torch.Tensor) -> torch.Tensor:
        """
        Calculate smoothness penalty.
        
        Args:
            w: (B, K) spline weights
            ddBx: (B, K) second derivatives of spline bases
        
        Returns:
            penalty: scalar, mean((d²y/dx²)²)
        """
        d2y_dx2 = self.curvature_calc.run(w, ddBx)  # (B,)
        return torch.mean(d2y_dx2 ** 2)