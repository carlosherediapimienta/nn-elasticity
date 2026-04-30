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
    - Per-product features (n x features): log-price, log-demand, obs mask,
      lags, rolling stats, missing indicators.

    Tensors are pre-stacked in __init__ so that __getitem__ returns 8 named
    tensors instead of ~161 scalar tensors. This reduces CPU->GPU transfers
    from ~161 small cudaMemcpy calls to 8 per batch.
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

    PER_PRODUCT_CAT_COLS = ["brand", "style"]  # categorical features

    def __init__(self, df: pd.DataFrame, n: int):
        self.n = n  # Number of products
        N = len(df) # Number of samples in the dataset

        # ── Store and week codes: stacked into a single (N, 2) long tensor ──────
        # Index 0 → store_code, index 1 → week_id.
        # After DataLoader collation: (B, 2).
        self.ids = torch.stack([
            torch.tensor(df["store_code"].values, dtype=torch.long),
            torch.tensor(df["week_id"].values,    dtype=torch.long),
        ], dim=1)

        # ── Time features (Fourier precomputed + week_rank): (N, 7) float ───────
        # Column order follows TIME_COLS, preserved for ProductTokenBuilder.
        # After DataLoader collation: (B, 7).
        self.time_feats = torch.stack([
            torch.tensor(df[col].values, dtype=torch.float32)
            for col in self.TIME_COLS
        ], dim=1)

        # ── Promo features: (N, 2) float ─────────────────────────────────────────
        # Column order follows PROMO_COLS.
        # After DataLoader collation: (B, 2).
        self.promo_feats = torch.stack([
            torch.tensor(df[col].values, dtype=torch.float32)
            for col in self.PROMO_COLS
        ], dim=1)

        # ── Log-prices per product: (N, n) float ─────────────────────────────────
        # prices[:, i] = log_price_i.
        # After DataLoader collation: (B, n). Used directly as x in ICDN.run().
        self.prices = torch.stack([
            torch.tensor(df[f"log_price_{i}"].values, dtype=torch.float32)
            for i in range(n)
        ], dim=1)

        # ── Log-demands per product: (N, n) float ────────────────────────────────
        # demands[:, i] = log_liters_i.
        # After DataLoader collation: (B, n). Used as y_true in the loss.
        self.demands = torch.stack([
            torch.tensor(df[f"log_liters_{i}"].values, dtype=torch.float32)
            for i in range(n)
        ], dim=1)

        # ── Observation mask per product: (N, n) float ───────────────────────────
        # obs_mask[:, i] = 1.0 if demand was observed for product i, else 0.0.
        # After DataLoader collation: (B, n). Used in the loss and weighted average.
        self.obs_mask = torch.stack([
            torch.tensor(df[f"obs_mask_{i}"].values, dtype=torch.float32)
            for i in range(n)
        ], dim=1)

        # ── Numerical features per product: (N, n, F) float ──────────────────────
        # per_prod_float[:, i, j] = PER_PRODUCT_COLS[j] for product i.
        # F = len(PER_PRODUCT_COLS). After DataLoader collation: (B, n, F).
        # Consumed in ProductTokenBuilder: per_prod_float[:, i, :] → (B, F).
        F = len(self.PER_PRODUCT_COLS)
        self.per_prod_float = torch.zeros(N, n, F, dtype=torch.float32)
        for i in range(n):
            for j, col in enumerate(self.PER_PRODUCT_COLS):
                self.per_prod_float[:, i, j] = torch.tensor(
                    df[f"{col}_{i}"].values, dtype=torch.float32
                )

        # ── Categorical features per product: (N, n, C) long ─────────────────────
        # per_prod_cat[:, i, 0] = brand_i, per_prod_cat[:, i, 1] = style_i.
        # C = len(PER_PRODUCT_CAT_COLS). After DataLoader collation: (B, n, C).
        # Missing columns default to 0 (padding_idx in the embeddings).
        C = len(self.PER_PRODUCT_CAT_COLS)
        self.per_prod_cat = torch.zeros(N, n, C, dtype=torch.long)
        for i in range(n):
            for j, col in enumerate(self.PER_PRODUCT_CAT_COLS):
                if f"{col}_{i}" in df.columns:
                    self.per_prod_cat[:, i, j] = torch.tensor(
                        df[f"{col}_{i}"].values, dtype=torch.long
                    )

        self._len = N  # Number of samples in the dataset

    def __len__(self) -> int:
        return self._len

    # In pytorch, in the training loop, we use the __getitem__ method to get the batch.
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Returns 8 named tensors per sample (pre-stacked in __init__).
        # The DataLoader collates these into 8 tensors with an added batch dimension,
        # eliminating ~153 redundant scalar tensors and their GPU transfer calls.
        return {
            "ids":            self.ids[idx],            # (2,)      long
            "time_feats":     self.time_feats[idx],     # (7,)      float
            "promo_feats":    self.promo_feats[idx],    # (2,)      float
            "prices":         self.prices[idx],         # (n,)      float
            "demands":        self.demands[idx],        # (n,)      float
            "obs_mask":       self.obs_mask[idx],       # (n,)      float
            "per_prod_float": self.per_prod_float[idx], # (n, F)    float
            "per_prod_cat":   self.per_prod_cat[idx],   # (n, C)    long
        }