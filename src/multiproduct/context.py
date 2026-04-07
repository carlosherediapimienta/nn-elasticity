import torch
import torch.nn as nn

# Precomputed time features in the dataset (Fourier + trend).
_TIME_COLS = [
    "week_rank",
    "sin_52", "cos_52",
    "sin_26", "cos_26",
    "sin_13", "cos_13",
] 

# Shared promo features by (store, week).
_PROMO_COLS = ["on_promo", "promo_intensity_store_week"] 

# Suffixes of columns by product: {suffix}_{i}
_PER_PRODUCT_COLS = [
    "lag_1", "lag_2", "lag_4",
    "roll_4", "roll_8", "roll_13",
    "miss_lag_1", "miss_lag_2", "miss_lag_4",
    "miss_roll_4", "miss_roll_8", "miss_roll_13",
    "weeks_seen_upc", "weeks_seen_store_upc",
    "liters_per_upc",
] 

class MultiProductContextEmbeddings(nn.Module):
    """
    Context embeddings for n-product wide format.
    Builds context vector from:
      - store embedding
      - precomputed Fourier time features + week_rank (from batch)
      - promo features
      - per-product lags, rolling means, missing indicators and static features

    Public API: forward(batch) -> (B, out_dim)
    """

    def __init__(
        self,
        n: int,
        n_stores: int,
        d_store: int = 24,
    ):
        """
        Args:
            n:        number of products
            n_stores: number of unique stores
            d_store:  store embedding dimension
        """
        super().__init__()
        self.n = n
        # Building the store embedding.
        # Recall that:
        # raw store_code:  101, 205, 312
        # ---- ColumnEncoder.factorize() ----
        # contiguous indices:   0,   1,   2     (what reaches the Dataset)
        # ---- nn.Embedding ----
        # row 0 → embedding of store 101
        # row 1 → embedding of store 205
        # row 2 → embedding of store 312
        self.emb_store = nn.Embedding(n_stores, d_store)

    @property
    def out_dim(self) -> int:
        return (
            self.emb_store.embedding_dim +  # d_store
            len(_TIME_COLS) +               # 7
            len(_PROMO_COLS) +              # 2
            len(_PER_PRODUCT_COLS) * self.n # 15 × n
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from MultiProductDataset.__getitem__(), containing:
                - "store_code": (B,) store index (label-encoded)
                - time features: (B,) each — week_rank, sin/cos Fourier components
                - promo features: (B,) each — on_promo, promo_intensity_store_week
                - per-product features: (B,) each — lags, rolling stats, 
                  missing indicators, static features

        Returns:
            c: (B, out_dim) context vector, where out_dim = d_store + 7 + 2 + 15*n
        """
        # store embedding: 1 tensor of shape (B, d_store)
        e_s = self.emb_store(batch["store_code"].long())

        # time features: 7 scalars → 7 tensors of shape (B, 1)
        time_feats = torch.stack(
            [batch[col] for col in _TIME_COLS], dim=1
        )

        # promo features: 2 scalars → 2 tensors of shape (B, 1)
        promo_feats = torch.stack(
            [batch[col] for col in _PROMO_COLS], dim=1
        )

        # per-product features: 15 × n scalars → n*15 tensors of shape (B, 1)
        per_product = []
        for i in range(self.n):
            for col in _PER_PRODUCT_COLS:
                per_product.append(batch[f"{col}_{i}"].unsqueeze(1)) # (B,) -> (B, 1)
        per_product_feats = torch.cat(per_product, dim=1)

        # Concatenate all the features.
        return torch.cat([e_s, time_feats, promo_feats, per_product_feats], dim=1) # (B, out_dim)