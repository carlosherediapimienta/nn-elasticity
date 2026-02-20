import torch
import torch.nn as nn
from src.nn.time_features import FourierTimeFeatures


class MultiProductContextEmbeddings(nn.Module):
    """
    Context embeddings for n-product wide format.
    No single UPC per row: store + week + Fourier + promo + n x per-product lags.
    Shared by nn and nn_baseline.
    Public API: forward(batch) -> (B, out_dim)
    """

    def __init__(
        self,
        n: int,
        n_stores: int,
        n_weeks: int,
        d_store: int = 24,
        d_week: int = 12,
        fourier_period: float = 52.0,
        fourier_harmonics: int = 4,
        include_trend: bool = True,
        week_min: float | None = None,
        week_max: float | None = None,
    ):
        """
        Args:
            n:               number of products
            n_stores:        number of unique stores
            n_weeks:         number of unique weeks
            d_store:         store embedding dimension
            d_week:          week embedding dimension
            fourier_period:  period for Fourier features (weeks)
            fourier_harmonics: number of Fourier harmonics
            include_trend:   include normalized trend scalar
            week_min/max:    range for trend normalization
        """
        super().__init__()
        self.n        = n
        self.week_min = week_min
        self.week_max = week_max

        self.emb_store = nn.Embedding(n_stores, d_store)
        self.emb_week  = nn.Embedding(n_weeks,  d_week)
        self.time_features = FourierTimeFeatures(
            period=fourier_period,
            harmonics=fourier_harmonics,
            include_trend=include_trend,
        )

    @property
    def out_dim(self) -> int:
        """Total context dimension."""
        return (
            self.emb_store.embedding_dim +   # store embedding
            self.emb_week.embedding_dim  +   # week embedding
            self.time_features.out_dim   +   # Fourier + trend
            4                            +   # on_promo, promo_B, promo_C, promo_S
            4 * self.n + 1                   # per-product lags + week_gap_1
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from MultiProductDataset.__getitem__()

        Returns:
            c: (B, out_dim) context vector
        """
        e_s = self.emb_store(batch["store_code"].long())
        e_w = self.emb_week(batch["week_id"].long())
        ft  = self.time_features(
            batch["week_id"],
            week_min=self.week_min,
            week_max=self.week_max,
        )
        promo = torch.stack(
            [batch["on_promo"], batch["promo_B"],
             batch["promo_C"], batch["promo_S"]], dim=1
        )

        lag_parts = []
        for i in range(self.n):
            lag_parts += [
                batch[f"lag_y_{i}_1"].unsqueeze(1),
                batch[f"rolling_mean_y_{i}_4"].unsqueeze(1),
                batch[f"lag_x_{i}_1"].unsqueeze(1),
                batch[f"delta_x_{i}_1"].unsqueeze(1),
            ]
        lag_parts.append(batch["week_gap_1"].unsqueeze(1))
        lags = torch.cat(lag_parts, dim=1)

        return torch.cat([e_s, e_w, ft, promo, lags], dim=1)