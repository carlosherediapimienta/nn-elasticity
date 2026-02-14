import math
import torch
import torch.nn as nn


class FourierTimeFeatures(nn.Module):
    """
    Fourier features for weekly time index.
    Returns [sin(2π k t / period), cos(2π k t / period)] for k=1..harmonics
    Optionally adds a normalized trend scalar.
    """
    def __init__(self, period: float = 52.0, harmonics: int = 4, include_trend: bool = True):
        super().__init__()
        self.period = float(period)
        self.harmonics = int(harmonics)
        self.include_trend = bool(include_trend)

    @property
    def out_dim(self) -> int:
        # 2 per harmonic + (optional) 1 trend
        return 2 * self.harmonics + (1 if self.include_trend else 0)

    def forward(self, week_id: torch.Tensor, week_min: float | None = None, week_max: float | None = None) -> torch.Tensor:
        """
        week_id: (B,) or (B,1) integer/float tensor
        week_min/week_max: if provided, trend = scaled to [-1,1]; else trend omitted or raw scaled.
        """
        t = week_id.float().view(-1, 1)  # (B,1)

        feats = []
        two_pi = 2.0 * math.pi
        for k in range(1, self.harmonics + 1):
            angle = two_pi * k * t / self.period
            feats.append(torch.sin(angle))
            feats.append(torch.cos(angle))

        if self.include_trend:
            if (week_min is not None) and (week_max is not None) and (week_max > week_min):
                # Scale to [-1, 1]
                trend = 2.0 * (t - week_min) / (week_max - week_min) - 1.0
            else:
                # Fallback: mild scaling to keep magnitude reasonable
                trend = (t - t.mean()) / (t.std(unbiased=False) + 1e-6)
            feats.append(trend)

        return torch.cat(feats, dim=1)  # (B, out_dim)