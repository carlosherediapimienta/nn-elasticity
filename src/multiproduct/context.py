import torch
import torch.nn as nn
from src.nn.time_features import FourierTimeFeatures

_ALL_REGRESSORS = frozenset({"lag_y", "lag_y_52", "rolling_mean_y", "week_gap"})


class MultiProductContextEmbeddings(nn.Module):
    """
    Context embeddings for n-product wide format.
    No single UPC per row: store + week + Fourier + promo + n x per-product lags.
    Public API: forward(batch) -> (B, out_dim)
    """

    def __init__(
        self,
        n: int,
        n_stores: int,
        d_store: int = 24,
        fourier_period: float = 52.0,
        fourier_harmonics: int = 4,
        include_trend: bool = True,
        week_min: float | None = None,
        week_max: float | None = None,
        regressors: frozenset[str] | set[str] = _ALL_REGRESSORS,
    ):
        """
        Args:
            n:               number of products
            n_stores:        number of unique stores
            d_store:         store embedding dimension
            fourier_period:  period for Fourier features (weeks)
            fourier_harmonics: number of Fourier harmonics
            include_trend:   include normalized trend scalar
            week_min/max:    range for trend normalization
            regressors:      set of regressors to include (default: all)
        """
        super().__init__()
        self.n        = n
        self.week_min = week_min
        self.week_max = week_max

        self.emb_store = nn.Embedding(n_stores, d_store)
        self.time_features = FourierTimeFeatures(
            period=fourier_period,
            harmonics=fourier_harmonics,
            include_trend=include_trend,
        )

        self.regressors = frozenset(regressors)

    @property
    def out_dim(self) -> int:
        _PER_PRODUCT = {"lag_y", "lag_y_52", "rolling_mean_y"}
        n_lag = sum(self.n for r in _PER_PRODUCT if r in self.regressors)
        n_gap = 1 if "week_gap" in self.regressors else 0
        return (
            self.emb_store.embedding_dim +
            self.time_features.out_dim   +
            4                            +   # promo
            n_lag + n_gap
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from MultiProductDataset.__getitem__()

        Returns:
            c: (B, out_dim) context vector
        """
        e_s = self.emb_store(batch["store_code"].long())
        ft  = self.time_features(
            batch["week_id"],
            week_min=self.week_min,
            week_max=self.week_max,
        )
        promo = torch.stack(
            [batch["on_promo"], batch["promo_B"],
             batch["promo_C"], batch["promo_S"]], dim=1
        )

        _REG_KEYS = {
            "lag_y":          lambda i: f"lag_y_{i}_1",
            "lag_y_52":       lambda i: f"lag_y_{i}_52",
            "rolling_mean_y": lambda i: f"rolling_mean_y_{i}_4",
        }

        lag_parts = []
        for i in range(self.n):
            for name, key_fn in _REG_KEYS.items():
                if name in self.regressors:
                    lag_parts.append(batch[key_fn(i)].unsqueeze(1))

        if "week_gap" in self.regressors:
            lag_parts.append(batch["week_gap_1"].unsqueeze(1))

        parts = [e_s, ft, promo]
        if lag_parts:
            parts.append(torch.cat(lag_parts, dim=1))
        return torch.cat(parts, dim=1)