import pandas as pd

class MultiProductPivoter:
    """
    Transforms a long-format multi-product panel into a wide-format table
    indexed by (store_code, week_id), aligned to `selected_upcs` order.
    Processing steps:
      1. Restrict input rows to `selected_upcs`.
      2. Build wide price matrix (`log_price_i`) from `log_price_per_liter`:
         - pivot by (store_code, week_id) x upc_code,
         - reindex columns to match `selected_upcs`,
         - within each store: forward-fill then backward-fill over weeks,
         - fill any remaining NaNs with the column mean.
      3. Build wide demand matrix (`log_liters_i`) from `log_liters_sold`:
         - pivot + reindex to `selected_upcs`,
         - create `obs_mask_i` = 1.0 where demand was observed, else 0.0,
         - fill missing demand values with 0.0.
      4. Add store-week shared features (`_STORE_WEEK_COLS`) using first value
         per (store, week), assuming consistency across UPC rows.
      5. Add optional per-UPC engineered features (`_PER_UPC_COLS`):
         - only columns present in `df` are included,
         - each is pivoted/reindexed/fillna(0.0) and renamed as `{short}_{i}`.
    Output:
      A wide DataFrame with one row per (store_code, week_id), containing:
        - `log_price_{i}`, `log_liters_{i}`, `obs_mask_{i}`
        - shared store-week features
        - optional per-UPC feature blocks (lags, rolling stats, missing flags,
          tenure-style features, etc.).
    Notes:
      - UPC index `i` is positional (0..n-1) and follows `selected_upcs`.
      - Missing demand is represented as 0.0 plus `obs_mask_i` to preserve
        observability information.
      - This method does not enforce panel completeness; filtering is expected
        upstream (e.g., via selection/filter components).
    Public API:
      - run(df, selected_upcs) -> pd.DataFrame
    """

    # columns that are equal for all UPCs in a (store, week) pair
    _STORE_WEEK_COLS = [
        "promo_intensity_store_week",
        "week_rank",
        "sin_52", "cos_52", "sin_26", "cos_26", "sin_13", "cos_13",
    ]

    # columns by UPC that will be pivoted to {name_i}
    _PER_UPC_COLS = [
        "on_promo",
        "lag_1_log_liters_sold",
        "lag_2_log_liters_sold",
        "lag_4_log_liters_sold",
        "rolling_mean_4_log_liters_sold",
        "rolling_mean_8_log_liters_sold",
        "rolling_mean_13_log_liters_sold",
        "miss_lag_1", "miss_lag_2", "miss_lag_4",
        "miss_roll_4", "miss_roll_8", "miss_roll_13",
        "weeks_since_first_seen_upc",
        "weeks_since_first_seen_store_upc",
        "liters_per_upc",
        # competitive features
        "n_neighbors_sw_cat",
        "neighbor_promo_share_sw_cat",
        "n_same_brand_neighbors_sw_cat",
        "same_brand_neighbor_promo_share_sw_cat",
        "lag1_neighbor_mean_log_liters_sold",
        "lag1_same_brand_neighbor_mean_log_liters_sold",
        "roll4_neighbor_mean_log_liters_sold",
        "miss_lag1_neighbor_mean_log_liters_sold",
        "miss_roll4_neighbor_mean_log_liters_sold",
        "miss_lag1_same_brand_neighbor_mean_log_liters_sold",
        "store_category_upc_count_static",
        "same_brand_upc_count_store_cat_static",
        "n_new_neighbors_13w",
        "share_new_neighbors_13w",
    ]

    # short names that will have in the wide format: col_{i}
    _PER_UPC_SHORT = [
        "on_promo",
        "lag_1", "lag_2", "lag_4",
        "roll_4", "roll_8", "roll_13",
        "miss_lag_1", "miss_lag_2", "miss_lag_4",
        "miss_roll_4", "miss_roll_8", "miss_roll_13",
        "weeks_seen_upc", "weeks_seen_store_upc", "liters_per_upc",
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
        "same_brand_upc_store_cat_count",
        "n_new_neighbors",
        "share_new_neighbors",
    ]

    _PER_UPC_CAT_COLS  = ["brand_family_norm", "style_segment_norm"]
    _PER_UPC_CAT_SHORT = ["brand", "style"]

    def _pivot_upc_col(
        self,
        df: pd.DataFrame,
        long_col: str,
        short_col: str,
        selected_upcs: list,
        n: int,
        aggfunc: str = "mean",
        fill_value: float | int = 0.0,
        as_int: bool = False,
    ) -> pd.DataFrame | None:

        if long_col not in df.columns:
            return None
        # Pivot the column to the wide format.
        pivoted = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values=long_col,
            aggfunc=aggfunc,
        ).reindex(columns=selected_upcs)
        # Fill missing values with the fill_value.
        pivoted = pivoted.fillna(fill_value)
        # If the column is an integer, convert it to an integer.
        if as_int:
            pivoted = pivoted.astype(int)
        # Rename the columns to {short_col}_{i}.
        pivoted.columns = [f"{short_col}_{i}" for i in range(n)]
        return pivoted

    def fit(self, df: pd.DataFrame, selected_upcs: list) -> "MultiProductPivoter":
        """Median and last observed log-price from TRAIN only."""
        price_obs = self._price_matrix(df, selected_upcs)
        self.selected_upcs_ = list(selected_upcs)
        self.median_price_ = price_obs.median(axis=0)  # skipna: observed only
        last = (
            price_obs.groupby(level="store_code")
            .ffill()
            .groupby(level="store_code")
            .last()
        )
        self.last_observed_ = last
        return self

    def transform(self, df: pd.DataFrame, selected_upcs: list | None = None, seed_from_train: bool = False) -> pd.DataFrame:
        selected_upcs = list(selected_upcs or self.selected_upcs_)
        n = len(selected_upcs)
        df = df[df["upc_code"].isin(selected_upcs)].copy()

        price_obs = self._price_matrix(df, selected_upcs)
        price_observed = price_obs.notna().astype(float)
        price_observed.columns = [f"price_observed_{i}" for i in range(n)]

        price_wide = self._ffill_causal(price_obs, use_seed=seed_from_train)
        if not hasattr(self, "median_price_"):
            raise RuntimeError("MultiProductPivoter.fit() on train before transform().")
        median = self.median_price_.reindex(price_wide.columns)
        price_wide = price_wide.fillna(median)
        price_wide.columns = [f"log_price_{i}" for i in range(n)]

        demand_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_liters_sold",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)

        obs_mask = demand_wide.notna().astype(float)
        availability = (
            price_obs.notna()
            | demand_wide.notna().reindex(price_obs.index).fillna(False)
        ).astype(float)

        demand_wide = demand_wide.fillna(0.0)
        demand_wide.columns = [f"log_liters_{i}" for i in range(n)]
        obs_mask.columns = [f"obs_mask_{i}" for i in range(n)]
        availability.columns = [f"availability_{i}" for i in range(n)]
        
        store_week_agg = (
            df.groupby(["store_code", "week_id"])[self._STORE_WEEK_COLS].first()
        )

        per_upc_parts = []
        for long_col, short_col in zip(self._PER_UPC_COLS, self._PER_UPC_SHORT):
            part = self._pivot_upc_col(df, long_col, short_col, selected_upcs, n)
            if part is not None:
                per_upc_parts.append(part)
        for long_col, short_col in zip(self._PER_UPC_CAT_COLS, self._PER_UPC_CAT_SHORT):
            part = self._pivot_upc_col(
                df, long_col, short_col, selected_upcs, n,
                aggfunc="first", fill_value=0, as_int=True,
            )
            if part is not None:
                per_upc_parts.append(part)

        wide = (
            price_wide.join(price_observed)
            .join(availability)
            .join(demand_wide)
            .join(obs_mask)
            .join(store_week_agg)
        )
        for part in per_upc_parts:
            wide = wide.join(part)
        return wide.reset_index()

    def _price_matrix(self, df: pd.DataFrame, selected_upcs: list) -> pd.DataFrame:
        price_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_price_per_liter",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)
        return (
            price_wide.reset_index()
            .sort_values(["store_code", "week_id"])
            .set_index(["store_code", "week_id"])
        )

    def _ffill_causal(self, price_obs, use_seed: bool = False):
        seeded = price_obs
        seed_week = None
        if use_seed and getattr(self, "last_observed_", None) is not None:
            stores = price_obs.index.get_level_values("store_code").unique()
            week0 = price_obs.index.get_level_values("week_id").min()
            seed_week = week0 - 1
            seed = self.last_observed_.reindex(stores)
            seed.index = pd.MultiIndex.from_product(
                [stores, [seed_week]], names=["store_code", "week_id"]
            )
            seeded = pd.concat([seed, price_obs]).sort_index()
        filled = seeded.groupby(level="store_code").ffill()
        if seed_week is not None:
            filled = filled[filled.index.get_level_values("week_id") != seed_week]
        return filled