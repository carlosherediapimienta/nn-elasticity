import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNaturalCubicSplineBasis(nn.Module):
    """Vectorized natural cubic spline basis (nonlinear part only) for n products.

    Alternative to MultiCubicSplineBasis: instead of a truncated power basis
    (unbounded cubic growth beyond the last knot), this basis imposes zero
    second derivative outside the boundary knots xi_1 and xi_{K+2}. Beyond
    that range each basis function is exactly linear, so the curvature and
    elasticity contributed by the spline stop growing artificially at the
    extremes of the training support.

    The explicit linear term beta_ii(x) * u_i of ICDN stays outside this
    module (in DemandParameterHead / DemandCalculator); this module only
    provides the K nonlinear basis functions N_1..N_K.

    Shapes:
    x    : (B, n)      log-price per (observation, product)
    Bx   : (B, n, K)   spline basis evaluated at x
    dBx  : (B, n, K)   first derivative  w.r.t. original x
    ddBx : (B, n, K)   second derivative w.r.t. original x

    Basis definition (product i, K+2 ordered knots xi_1 < ... < xi_{K+2},
    where xi_1 and xi_{K+2} are the boundary knots = min/max of the training
    support and xi_2..xi_{K+1} are the K interior knots):

    xs      = (x - shift_i) / scale_i
    D_j(xs) = [ReLU(xs - xi_j)^3 - ReLU(xs - xi_{K+2})^3] / (xi_{K+2} - xi_j),  j = 1..K+1
    N_j(xs) = D_j(xs) - D_{K+1}(xs),                                          j = 1..K

    All derivatives and the basis are corrected via the chain rule to be
    expressed w.r.t. the original (un-normalized) price x, same convention
    as MultiCubicSplineBasis.
    """

    def __init__(
        self,
        knots: torch.Tensor,  # (n, K+2) — boundary + interior knots, strictly increasing
        shift: torch.Tensor,  # (n,)
        scale: torch.Tensor,  # (n,)
    ):
        super().__init__()
        if knots.ndim != 2:
            raise ValueError(f"knots must have shape (n, K+2). Got {tuple(knots.shape)}")
        n, K_total = knots.shape
        if K_total < 3:
            raise ValueError(
                "knots must contain 2 boundary knots plus at least 1 interior "
                f"knot (K+2 >= 3). Got K_total={K_total}"
            )

        shift = shift.view(n).float()
        scale = scale.view(n).float()
        if torch.any(scale <= 0):
            raise ValueError("scale must be > 0 for all products")

        knots = knots.float()
        if torch.any(knots[:, 1:] <= knots[:, :-1]):
            raise ValueError("knots must be strictly increasing per product")

        # Pre-scale knots so forward() can be a single broadcasted op,
        # same convention as MultiCubicSplineBasis.
        knots_s = (knots - shift[:, None]) / scale[:, None]  # (n, K+2)

        # xi_{K+2} - xi_j for j = 1..K+1 (all knots except the right boundary
        # itself, which is only used as the reference point). Guaranteed > 0
        # by the strictly-increasing check above -> no null denominators.
        right = knots_s[:, -1:]      # (n, 1)   xi_{K+2}
        left = knots_s[:, :-1]       # (n, K+1) xi_1 .. xi_{K+1}
        denom = right - left          # (n, K+1)

        self.register_buffer("knots_left", left)    # (n, K+1)
        self.register_buffer("right_knot", right)   # (n, 1)
        self.register_buffer("denom", denom)         # (n, K+1)
        self.register_buffer("shift", shift)         # (n,)
        self.register_buffer("scale", scale)         # (n,)
        self._K = K_total - 2

    @property
    def n(self) -> int:
        return int(self.knots_left.shape[0])

    @property
    def K(self) -> int:
        """Dimension of the nonlinear basis (number of interior knots)."""
        return int(self._K)

    def forward(self, x: torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"x must have shape (B, n). Got {tuple(x.shape)}")
        B, n = x.shape
        if n != self.n:
            raise ValueError(f"x has n={n}, but spline was built for n={self.n}")

        x = x.float()
        xs = (x - self.shift[None, :]) / self.scale[None, :]     # (B, n)
        xs = xs[:, :, None]                                       # (B, n, 1)

        u_left = F.relu(xs - self.knots_left[None, :, :])         # (B, n, K+1)
        u_right = F.relu(xs - self.right_knot[None, :, :])        # (B, n, 1)
        inv_den = 1.0 / self.denom[None, :, :]                    # (1, n, K+1)

        # ───── THEORY IMPLEMENTATION: D_j, D'_j, D''_j ─────────────────────
        D_s = (u_left ** 3 - u_right ** 3) * inv_den                # (B, n, K+1)
        dD_s = (3.0 * u_left ** 2 - 3.0 * u_right ** 2) * inv_den    # (B, n, K+1)
        ddD_s = (6.0 * u_left - 6.0 * u_right) * inv_den              # (B, n, K+1)

        # N_j = D_j - D_{K+1}: the last column of D_s/dD_s/ddD_s is D_{K+1};
        # this subtraction is what cancels the linear growth beyond xi_{K+2}.
        Bx_s = D_s[..., :-1] - D_s[..., -1:]
        dBx_s = dD_s[..., :-1] - dD_s[..., -1:]
        ddBx_s = ddD_s[..., :-1] - ddD_s[..., -1:]

        inv_scale = (1.0 / self.scale)[None, :, None]                # (1, n, 1)
        Bx = Bx_s
        dBx = dBx_s * inv_scale
        ddBx = ddBx_s * (inv_scale ** 2)
        return Bx, dBx, ddBx