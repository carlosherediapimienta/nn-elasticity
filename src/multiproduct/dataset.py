import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiProductDataset(Dataset):
    """
    Dataset for n-product wide format (output of MultiProductBuilder).
    Each sample: (store, week) with shared time features, promo features,
    n prices, n demands, n obs_masks, and per-product precomputed lags,
    rolling means and missing indicators.
    Public API: __len__(), __getitem__()
    """

    # columnas de tiempo compartidas por (store, week) — Fourier precomputado + trend
    TIME_COLS = [
        "week_rank",
        "sin_52", "cos_52",
        "sin_26", "cos_26",
        "sin_13", "cos_13",
    ]

    # columnas de promo compartidas por (store, week)
    PROMO_COLS = ["on_promo", "promo_intensity_store_week"]

    # sufijos de columnas por producto i → {sufijo}_{i}
    PER_PRODUCT_COLS = [
        "lag_1", "lag_2", "lag_4",
        "roll_4", "roll_8", "roll_13",
        "miss_lag_1", "miss_lag_2", "miss_lag_4",
        "miss_roll_4", "miss_roll_8", "miss_roll_13",
        "weeks_seen_upc", "weeks_seen_store_upc",
        "liters_per_upc",
    ]

    def __init__(self, df: pd.DataFrame, n: int):
        self.n = n
        t = {}

        # ── identificadores ───────────────────────────────────────────────
        t["store_code"] = torch.tensor(df["store_code"].values, dtype=torch.long)
        t["week_id"]    = torch.tensor(df["week_id"].values,    dtype=torch.long)

        # ── features de tiempo (Fourier precomputado + week_rank) ─────────
        for col in self.TIME_COLS:
            t[col] = torch.tensor(df[col].values, dtype=torch.float32)

        # ── features de promo ─────────────────────────────────────────────
        for col in self.PROMO_COLS:
            t[col] = torch.tensor(df[col].values, dtype=torch.float32)

        # ── features por producto ─────────────────────────────────────────
        for i in range(n):
            t[f"log_price_{i}"]  = torch.tensor(df[f"log_price_{i}"].values,  dtype=torch.float32)
            t[f"log_liters_{i}"] = torch.tensor(df[f"log_liters_{i}"].values, dtype=torch.float32)
            t[f"obs_mask_{i}"]   = torch.tensor(df[f"obs_mask_{i}"].values,   dtype=torch.float32)

            for col in self.PER_PRODUCT_COLS:
                t[f"{col}_{i}"] = torch.tensor(df[f"{col}_{i}"].values, dtype=torch.float32)

        self.tensors = t
        self._len    = len(df)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}