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
_PROMO_COLS = ["promo_intensity_store_week"] 

# Suffixes of columns by product: {suffix}_{i}
_PER_PRODUCT_COLS = [
    "on_promo",
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

class ProductTokenBuilder(nn.Module):
    """
    Context embeddings for n-product wide format.
    Builds context token for each product from:
      - store embedding
      - precomputed Fourier time features + week_rank (from batch)
      - promo features
      - per-product lags, rolling means, missing indicators and static features
    
    The context token is a tensor of shape (B, n, d_token) where:
    - B is the batch size.
    - n is the number of products.
    - d_token is the dimension of the context token.

    Each token_i = [global_broadcast | own_features_i]

    Public API: forward(batch) -> (B, n, d_token)
    """

    def __init__(
        self,
        n: int,
        n_stores: int,
        d_store: int = 24,
        n_brands: int = 1,
        d_brand: int = 8,
        n_styles: int = 1,
        d_style: int = 8,
    ):
        """
        Args:
            n:        number of products
            n_stores: number of unique stores
            d_store:  store embedding dimension
        """
        super().__init__()
        self.n = n
        # Building the store embedding and the brand and style embeddings.
        # Recall that:
        # raw store_code:  101, 205, 312
        # ---- ColumnEncoder.factorize() ----
        # contiguous indices:   0,   1,   2     (what reaches the Dataset)
        # ---- nn.Embedding ----
        # row 0 → embedding of store 101
        # row 1 → embedding of store 205
        # row 2 → embedding of store 312
        self.emb_store = nn.Embedding(n_stores, d_store)
        # Building the brand and style embeddings.
        # Recall that: 
        # +1 is to preserve the 0 index for the unknown brand and style.
        # padding_idx=0 is to map this token to the padding.
        self.emb_brand = nn.Embedding(n_brands + 1, d_brand, padding_idx=0)
        self.emb_style = nn.Embedding(n_styles + 1, d_style, padding_idx=0)
        self.d_brand   = d_brand
        self.d_style   = d_style

    @property
    def d_token(self) -> int:
        return (
            self.emb_store.embedding_dim +  # d_store
            len(_TIME_COLS) +               # 7
            len(_PROMO_COLS) +              # 2
            self.d_brand +                  # d_brand
            self.d_style +                  # d_style
            len(_PER_PRODUCT_COLS)          # 25 (15 + 10 competitive)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict from MultiProductDataset.__getitem__(), containing:
                - "ids":            (B, 2)    long  — [:, 0]=store_code, [:, 1]=week_id
                - "time_feats":     (B, 7)    float — Fourier precomputed + week_rank
                - "promo_feats":    (B, 2)    float — on_promo, promo_intensity_store_week
                - "per_prod_float": (B, n, F) float — lags, rolling stats, missing indicators
                - "per_prod_cat":   (B, n, C) long  — [:, i, 0]=brand_i, [:, i, 1]=style_i
        Returns:
            c: (B, n, d_token) context tensor, one token per product per sample.
        """
        # store embedding: (B, d_store).
        # ids[:, 0] holds the store_code (label-encoded contiguous index).
        e_s = self.emb_store(batch["ids"][:, 0])
        # ---Global broadcast: the same for all n tokens of each observation.
        # time_feats and promo_feats are already (B, 7) and (B, 2) — no stack needed.
        global_ctx = torch.cat(
            [e_s, batch["time_feats"], batch["promo_feats"]], dim=1
        )  # (B, d_store + 7 + 2)
        # tokens per-product
        tokens = []
        for i in range(self.n):
            # per_prod_cat[:, i, 0] = brand_i  — .long() ensures integer dtype for embedding lookup.
            e_brand = self.emb_brand(batch["per_prod_cat"][:, i, 0])  # (B, d_brand)
            # per_prod_cat[:, i, 1] = style_i
            e_style = self.emb_style(batch["per_prod_cat"][:, i, 1])  # (B, d_style)
            # per_prod_float[:, i, :] already has shape (B, F) — no stack needed.
            per_i   = batch["per_prod_float"][:, i, :]                # (B, F)
            token_i = torch.cat([global_ctx, e_brand, e_style, per_i], dim=1)
            tokens.append(token_i)
        # Stack the per-product tokens along the product dimension.
        return torch.stack(tokens, dim=1)  # (B, n, d_token)