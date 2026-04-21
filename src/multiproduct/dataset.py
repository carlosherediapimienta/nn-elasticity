import pandas as pd
import torch
from torch.utils.data import Dataset

# Notice that this class there is no Public API with a return.
# Recall that the __getitem__ method is used to get the batch in the training loop.
class MultiProductDataset(Dataset):
    """
    PyTorch Dataset for the n-product wide format produced by MultiProductBuilder.

    Each sample corresponds to one (store, week) observation and contains:
    - Shared features: Fourier time encodings, week rank, promo indicators.
    - Per-product features (n x features): log-price, log-demand, obs mask, lags, rolling stats, missing indicators.
    """

    # Shared time features by (store, week): Fourier precomputed + trend
    TIME_COLS = [
        "week_rank",
        "sin_52", "cos_52",
        "sin_26", "cos_26",
        "sin_13", "cos_13",
    ]

    # Shared promo features by (store, week)
    PROMO_COLS = ["on_promo", "promo_intensity_store_week"]

    # Per-product suffixes: {suffix}_{i}
    PER_PRODUCT_COLS = [
        "lag_1", "lag_2", "lag_4",
        "roll_4", "roll_13",
        "miss_lag_1", "miss_lag_2", "miss_lag_4",
        "miss_roll_4", "miss_roll_13",
        "weeks_seen_upc", "weeks_seen_store_upc",
        "liters_per_upc",
        # competitive features
        "n_neighbors",
        "nb_promo_share",
        "n_same_brand_neighbors",
        "sb_promo_share",
        "lag1_nb_mean_demand",
        "lag1_sb_mean_demand",
        "roll4_nb_mean_demand",
        "miss_lag1_nb_demand",
        "miss_roll4_nb_demand",
        "miss_lag1_sb_demand",
        "store_cat_upc_count",
        "n_new_neighbors",
        "share_new_neighbors",
    ]

    PER_PRODUCT_CAT_COLS = ["brand", "style"] # categorical features

    def __init__(self, df: pd.DataFrame, n: int):
        self.n = n # Number of products
        t = {} # Dictionary to store the tensors

        # ── Store and week codes ───────────────────────────────────────────────
        t["store_code"] = torch.tensor(df["store_code"].values, dtype=torch.long)
        t["week_id"]    = torch.tensor(df["week_id"].values,    dtype=torch.long)

        # ── Time features (Fourier precomputed + week_rank) ─────────
        for col in self.TIME_COLS:
            t[col] = torch.tensor(df[col].values, dtype=torch.float32)

        # ── Promo features ─────────────────────────────────────────────
        for col in self.PROMO_COLS:
            t[col] = torch.tensor(df[col].values, dtype=torch.float32)

        # ── Features per product ─────────────────────────────────────────
        for i in range(self.n):
            t[f"log_price_{i}"]  = torch.tensor(df[f"log_price_{i}"].values,  dtype=torch.float32)
            t[f"log_liters_{i}"] = torch.tensor(df[f"log_liters_{i}"].values, dtype=torch.float32)
            t[f"obs_mask_{i}"]   = torch.tensor(df[f"obs_mask_{i}"].values,   dtype=torch.float32)
            for col in self.PER_PRODUCT_COLS:
                t[f"{col}_{i}"] = torch.tensor(df[f"{col}_{i}"].values, dtype=torch.float32)
            for col in self.PER_PRODUCT_CAT_COLS:
                if f"{col}_{i}" in df.columns:
                    t[f"{col}_{i}"] = torch.tensor(df[f"{col}_{i}"].values, dtype=torch.long)

        self.tensors = t # Dictionary with the tensors
        self._len    = len(df) # Number of samples in the dataset

    def __len__(self) -> int:
        return self._len

    # In pytorch, in the training loop, we use the __getitem__ method to get the batch.
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}