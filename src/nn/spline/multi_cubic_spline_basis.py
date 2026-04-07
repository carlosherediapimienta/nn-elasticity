import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiCubicSplineBasis(nn.Module):
    """Vectorized cubic truncated power spline basis for n products.

    Accelerates the common case where all products share the same number of
    knots (K), but each product has its own knot locations and price
    normalization (shift/scale). All products are processed in a single
    batched operation instead of a per-product loop.

    Shapes:
    x    : (B, n)      log-price per (observation, product)
    Bx   : (B, n, K)   spline basis evaluated at x
    dBx  : (B, n, K)   first derivative  w.r.t. original x  (proportional to elasticity)
    ddBx : (B, n, K)   second derivative w.r.t. original x  (smoothness penalty)
    dddBx: (B, n, K)   third derivative  w.r.t. original x  (smoothness penalty)
    IBx  : (B, n, K)   antiderivative    w.r.t. original x  (integrability constraint)

    Basis definition (product i, knot k):
    xs        = (x - shift_i) / scale_i          # normalize price to ~N(0,1)
    u_{i,k}   = ReLU(xs - knot_{i,k})            # truncated at knot location
    B_{i,k}   = u_{i,k}^3                        # cubic truncated power basis

    All derivatives and the antiderivative are corrected via the chain rule
    to be expressed w.r.t. the original (un-normalized) price x.
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

        # Convert the shift and scale to a float tensor. 
        # The view() method is used to reshape (in case it is not already) 
        # the tensor into a 1D tensor of shape (n,).
        shift = shift.view(n).float() # (n,)
        scale = scale.view(n).float() # (n,)
        if torch.any(scale <= 0): # If the scale is not greater than 0, raise an error.
            raise ValueError("scale must be > 0 for all products")

        # Pre-scale knots so forward() can be a single broadcasted op.
        # Recall that:
        # - shift is the mean of the data for each product.
        # - scale is the standard deviation of the data for each product.
        knots_s = (knots.float() - shift[:, None]) / scale[:, None]

        # We register the pre-scaled knots, shift and scale 
        # as buffers. This means that they will be treated as constants 
        # and will be moved to the same device as the model.
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
        # We verify that x has the same dimensions as 
        # the number of products as the spline was built for.
        if n != self.n:
            raise ValueError(f"x has n={n}, but spline was built for n={self.n}")

        x = x.float()
        # The self.shift,self.scale and self.knots are buffers,
        # so they are already on the same device as the model.
        # We normalize the price for each product.
        xs = (x - self.shift[None, :]) / self.scale[None, :]          # (B, n)
        # We truncate the price at the knots for each product.
        # Let us put an example:
        # Let us say that we have 1 product with 3 knots [0.2, 0.5, 0.8]
        # u = ReLU(xs - knots)
        # u = ReLU([0.6 - 0.2,  0.6 - 0.5,  0.6 - 0.8])
        # u = ReLU([  0.4,         0.1,        -0.2   ])
        # u =      [  0.4,         0.1,         0.0   ]
        # so, Bx = [0.064,  0.001,  0.0]
        # and consquently, curve(xs) = w_1 Bx_1 + w_2 Bx_2 + w_3 Bx_3
        # where each weight is a learnable parameter of the model, and the 
        # Neural Network can independently fit a curve in each section of the price.
        u = F.relu(xs[:, :, None] - self.knots[None, :, :])           # (B, n, K)

        # ───── THEORY IMPLEMENTATION: See Article ────────────────────────────────────
        Bx_s = u ** 3
        dBx_s = 3.0 * (u ** 2)
        ddBx_s = 6.0 * u
        dddBx_s = 6.0 * (u > 0).float()
        IBx_s = (u ** 4) / 4.0
        
        # We recover the original price from the normalized price.
        inv_scale = (1.0 / self.scale)[None, :, None]                # (1, n, 1)
        Bx = Bx_s
        dBx = dBx_s * inv_scale
        ddBx = ddBx_s * (inv_scale ** 2)
        dddBx = dddBx_s * (inv_scale ** 3)
        IBx = IBx_s * self.scale[None, :, None]
        return Bx, dBx, ddBx, dddBx, IBx