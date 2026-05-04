import torch
import torch.nn as nn
import torch.nn.functional as F


class CubicSplineBasis(nn.Module):
    """
    Cubic regression spline basis using truncated power functions:
      B_k(x) = ReLU(x - knot_k)^3
    
    Smooth up to the second derivative, and we can calculate B'(x), B''(x) analytically.
    
    Optional scaling of x for numerical stability:
      x_s = (x - shift) / scale
    In that case, dB/dx = (dB/dx_s) * (1/scale)
    """
    
    def __init__(self, knots: torch.Tensor, shift: float = 0.0, scale: float = 1.0):
        """
        Args:
            knots: tensor (K,) with positions of the knots
            shift: shift for normalization
            scale: scale for normalization
        """
        super().__init__()
        knots = (knots.float() - shift) / scale 
        self.register_buffer("knots", knots.view(1, -1))
        self.shift = float(shift)
        self.scale = float(scale)

    @property
    def K(self) -> int:
        """Number of knots (dimension of the basis)."""
        return self.knots.shape[1]

    def forward(self, x: torch.Tensor):
        xs = x.float().view(-1, 1)              # (B, 1)
        xs = (xs - self.shift) / self.scale     # escalado
        u  = F.relu(xs - self.knots)            # (B, K)  ← knots ya están pre-escalados en __init__

        Bx_s   = u ** 3
        dBx_s  = 3.0 * (u ** 2)
        ddBx_s = 6.0 * u     

        inv_scale = 1.0 / self.scale
        Bx   = Bx_s
        dBx  = dBx_s  * inv_scale
        ddBx = ddBx_s * (inv_scale ** 2)
        return Bx, dBx, ddBx                    # (B, K)