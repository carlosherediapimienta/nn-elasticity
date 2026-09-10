import torch
import torch.nn as nn

from .multi_cubic_spline_basis import MultiCubicSplineBasis
from .multi_natural_cubic_spline_basis import MultiNaturalCubicSplineBasis


def build_price_basis(
    basis_type: str,
    spline_configs: list[dict],
) -> nn.Module:
    """
    Centralized factory for the price-basis module consumed by ICDN.

    Args:
        basis_type: "truncated_cubic" or "natural_cubic".
        spline_configs: one dict per product, as returned by
            SplineBuilder.build_from_data() — each with keys
            "knots" ((K,) or (K+2,) depending on basis_type), "mean", "std".

    Returns:
        nn.Module exposing forward(x) -> (Bx, dBx, ddBx), each (B, n, K).
    """
    knots = torch.stack([cfg["knots"] for cfg in spline_configs], dim=0)
    shift = torch.tensor([cfg["mean"] for cfg in spline_configs])
    scale = torch.tensor([cfg["std"] for cfg in spline_configs])

    if basis_type == "truncated_cubic":
        return MultiCubicSplineBasis(knots=knots, shift=shift, scale=scale)
    if basis_type == "natural_cubic":
        return MultiNaturalCubicSplineBasis(knots=knots, shift=shift, scale=scale)
    raise ValueError(f"Unknown basis_type: {basis_type!r}")