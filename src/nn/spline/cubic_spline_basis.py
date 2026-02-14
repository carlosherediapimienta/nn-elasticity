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
        knots = knots.float().view(1, -1)  # (1, K)
        self.register_buffer("knots", knots)
        self.shift = float(shift)
        self.scale = float(scale)

    @property
    def K(self) -> int:
        """Number of knots (dimension of the basis)."""
        return self.knots.shape[1]

    def _x_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """Scale x for numerical stability."""
        x = x.float().view(-1, 1)  # (B,1)
        return (x - self.shift) / self.scale

    def forward(self, x: torch.Tensor):
        """
        Calculate basis spline and its derivatives.
        
        Args:
            x: (B,) or (B,1) tensor with input values
        
        Returns:
            Bx: (B, K) values of the basis
            dBx: (B, K) first derivative w.r.t. x ORIGINAL
            ddBx: (B, K) second derivative w.r.t. x ORIGINAL
        """
        xs = self._x_scaled(x)                      # (B,1) scaled
        u = F.relu(xs - self.knots)                 # (B,K)

        Bx_s = u**3                                 # (B,K) in scaled space
        dBx_s = 3.0 * (u**2)                        # d/dx_s
        ddBx_s = 6.0 * u                            # d2/dx_s2

        # chain rule back to ORIGINAL x
        inv_scale = 1.0 / self.scale
        Bx = Bx_s
        dBx = dBx_s * inv_scale
        ddBx = ddBx_s * (inv_scale**2)

        return Bx, dBx, ddBx