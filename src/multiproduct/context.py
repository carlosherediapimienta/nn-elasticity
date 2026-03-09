import torch
import torch.nn as nn


# columnas de tiempo precomputadas en el dataset (Fourier + trend)
_TIME_COLS = [
    "week_rank",
    "sin_52", "cos_52",
    "sin_26", "cos_26",
    "sin_13", "cos_13",
]  # → 7 escalares

# columnas de promo compartidas por (store, week)
_PROMO_COLS = ["on_promo", "promo_intensity_store_week"]  # → 2 escalares

# sufijos de columnas por producto → {sufijo}_{i}
_PER_PRODUCT_COLS = [
    "lag_1", "lag_2", "lag_4",
    "roll_4", "roll_8", "roll_13",
    "miss_lag_1", "miss_lag_2", "miss_lag_4",
    "miss_roll_4", "miss_roll_8", "miss_roll_13",
    "weeks_seen_upc", "weeks_seen_store_upc",
    "liters_per_upc",
]  # → 15 escalares × n productos


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
            batch: dict from MultiProductDataset.__getitem__()

        Returns:
            c: (B, out_dim) context vector
        """
        # store embedding
        e_s = self.emb_store(batch["store_code"].long())

        # features de tiempo: 7 escalares → (B, 7)
        time_feats = torch.stack(
            [batch[col] for col in _TIME_COLS], dim=1
        )

        # features de promo: 2 escalares → (B, 2)
        promo_feats = torch.stack(
            [batch[col] for col in _PROMO_COLS], dim=1
        )

        # features por producto: 15 × n escalares → (B, 15*n)
        per_product = []
        for i in range(self.n):
            for col in _PER_PRODUCT_COLS:
                per_product.append(batch[f"{col}_{i}"].unsqueeze(1))
        per_product_feats = torch.cat(per_product, dim=1)

        return torch.cat([e_s, time_feats, promo_feats, per_product_feats], dim=1)