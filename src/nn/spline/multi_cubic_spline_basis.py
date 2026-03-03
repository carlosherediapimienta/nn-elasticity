import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCubicSplineBasis(nn.Module):
    """Vectorized cubic spline basis for multiple products.

    This is a drop-in *accelerator* for the common case where all products use
    the same number of knots (K), but each product can have its own knot
    locations and normalization (shift/scale).

    Inputs/outputs:
      x:   (B, n)
      Bx:  (B, n, K)
      dBx: (B, n, K)  derivative wrt *original* x
      ddBx:(B, n, K)
      IBx: (B, n, K)  antiderivative wrt original x

    Basis definition (per product i, knot k):
      xs = (x - shift_i) / scale_i
      u  = ReLU(xs - knot_{i,k})
      B  = u^3

    With the chain rule corrections so that derivatives are w.r.t. the original
    (un-normalized) x.
    """

    def __init__(
        self,
        knots: torch.Tensor,  # (n, K)
        shift: torch.Tensor,  # (n,)
        scale: torch.Tensor,  # (n,)
    ):
        super().__init__()
        if knots.ndim != 2:
            raise ValueError(f"knots must have shape (n, K). Got {tuple(knots.shape)}")
        n, K = knots.shape

        shift = shift.view(n).float()
        scale = scale.view(n).float()
        if torch.any(scale <= 0):
            raise ValueError("scale must be > 0 for all products")

        # Pre-scale knots so forward() can be a single broadcasted op.
        knots_s = (knots.float() - shift[:, None]) / scale[:, None]

        self.register_buffer("knots", knots_s)   # (n, K)
        self.register_buffer("shift", shift)     # (n,)
        self.register_buffer("scale", scale)     # (n,)

    @property
    def n(self) -> int:
        return int(self.knots.shape[0])

    @property
    def K(self) -> int:
        return int(self.knots.shape[1])

    def forward(self, x: torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"x must have shape (B, n). Got {tuple(x.shape)}")
        B, n = x.shape
        if n != self.n:
            raise ValueError(f"x has n={n}, but spline was built for n={self.n}")

        x = x.float()
        xs = (x - self.shift[None, :]) / self.scale[None, :]          # (B, n)
        u = F.relu(xs[:, :, None] - self.knots[None, :, :])           # (B, n, K)

        Bx_s = u ** 3
        dBx_s = 3.0 * (u ** 2)
        ddBx_s = 6.0 * u
        IBx_s = (u ** 4) / 4.0

        inv_scale = (1.0 / self.scale)[None, :, None]                # (1, n, 1)
        Bx = Bx_s
        dBx = dBx_s * inv_scale
        ddBx = ddBx_s * (inv_scale ** 2)
        IBx = IBx_s * self.scale[None, :, None]
        return Bx, dBx, ddBx, IBx