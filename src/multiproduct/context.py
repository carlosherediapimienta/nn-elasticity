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
                - "store_code": (B,) store index (label-encoded)
                - time features: (B,) each — week_rank, sin/cos Fourier components
                - promo features: (B,) each — on_promo, promo_intensity_store_week
                - per-product features: (B,) each — lags, rolling stats, 
                  missing indicators, static features

        Returns:
            c: (B, out_dim) context vector, where out_dim = d_store + 7 + 2 + 15
        """
        # store embedding: 1 tensor of shape (B, d_store)
        B = batch["store_code"].shape[0]

        # ---Global broadcast: the same for all n tokens of each observation
        e_s = self.emb_store(batch["store_code"].long())
        # time features: 7 scalars to a tensor of shape (B, len(_TIME_COLS)) or (B, 7) for example
        time_feats = torch.stack(
            [batch[col] for col in _TIME_COLS], dim=1
        )
        # promo features: 2 scalars to a tensor of shape (B, len(_PROMO_COLS)) or (B, 2) for example
        promo_feats = torch.stack(
            [batch[col] for col in _PROMO_COLS], dim=1
        )
        global_ctx = torch.cat([e_s, time_feats, promo_feats], dim=1) # (B, d_store + 7 + 2)

        # tokens per-product
        tokens = []
        for i in range(self.n):
            # .long() is to convert the brand and style to integers (not floats).
            e_brand = self.emb_brand(batch[f"brand_{i}"].long())  # (B, d_brand)
            e_style = self.emb_style(batch[f"style_{i}"].long())  # (B, d_style)
            per_i = torch.stack(
                [batch[f"{col}_{i}"] for col in _PER_PRODUCT_COLS], dim=1
            ) # (B, len(_PER_PRODUCT_COLS)) or (B, 15) for example
            token_i = torch.cat([global_ctx, e_brand, e_style, per_i], dim=1)
            tokens.append(token_i)

        # Concatenate all the features.
        return torch.stack(tokens, dim=1) # (B, n, d_token)