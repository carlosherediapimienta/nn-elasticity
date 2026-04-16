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
        "on_promo", "promo_intensity_store_week",
        "week_rank",
        "sin_52", "cos_52", "sin_26", "cos_26", "sin_13", "cos_13",
    ]

    # columns by UPC that will be pivoted to {name_i}
    _PER_UPC_COLS = [
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
        "n_active_neighbors_sw_cat",
        "neighbor_weighted_mean_log_price",
        "neighbor_min_log_price",
        "neighbor_promo_share",
        "price_gap_to_neighbors_mean",
        "price_gap_to_cheapest_neighbor",
        "miss_neighbor_weighted_mean_log_price",
        "miss_neighbor_min_log_price",
        "miss_price_gap_to_neighbors_mean",
        "miss_price_gap_to_cheapest_neighbor",
    ]

    # short names that will have in the wide format: col_{i}
    _PER_UPC_SHORT = [
        "lag_1", "lag_2", "lag_4",
        "roll_4", "roll_8", "roll_13",
        "miss_lag_1", "miss_lag_2", "miss_lag_4",
        "miss_roll_4", "miss_roll_8", "miss_roll_13",
        "weeks_seen_upc", "weeks_seen_store_upc", "liters_per_upc",
        # competitive features
        "n_active_neighbors",
        "nb_wmean_price",
        "nb_min_price",
        "nb_promo_share",
        "price_gap_mean",
        "price_gap_cheap",
        "miss_nb_wmean_price",
        "miss_nb_min_price",
        "miss_price_gap_mean",
        "miss_price_gap_cheap",
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

    def run(self, df: pd.DataFrame, selected_upcs: list) -> pd.DataFrame:
        df = df[df["upc_code"].isin(selected_upcs)].copy()
        n = len(selected_upcs)

        # price pivot table: (store_code, week_id) x {upc_code}
        # Example
        # rows: (store_code, week_id, upc_code, log_price_per_liter)
        # to 
        # row (store_code=A, week_id=10) -> columns [upc_code=1, upc_code=2, upc_code=3, ...] with their log_price_per_liter.
        price_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_price_per_liter",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)

        price_wide = (
            price_wide
            .reset_index()
            .sort_values(["store_code", "week_id"])
            .set_index(["store_code", "week_id"])
        )

        # It can be possible that there are gaps in the prices because the UPC was not available in some weeks.
        # This is why we need to forward-fill and backward-fill the prices over the weeks.
        price_wide = (
            price_wide
            .groupby("store_code", group_keys=False)
            .apply(lambda g: g.ffill().bfill())
        )
        for col in price_wide.columns: # Fill missing prices with the column mean
            price_wide[col] = price_wide[col].fillna(price_wide[col].mean())
        price_wide.columns = [f"log_price_{i}" for i in range(n)] # Rename the columns to log_price_{i}

        # demand pivot table + obs_mask: (store_code, week_id) x {upc_code}
        # Example
        # rows: (store_code, week_id, upc_code, log_liters_sold)
        # to 
        # row (store_code=A, week_id=10) -> columns [upc_code=1, upc_code=2, upc_code=3, ...] with their log_liters_sold.
        demand_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_liters_sold",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)

        # Create the obs_mask: 1.0 where demand was observed, else 0.0.
        obs_mask = demand_wide.notna().astype(float)
        # Because we already have the obs_mask
        # (we have already kept this information in the obs_mask), we can fill missing demand with 0.0.
        demand_wide = demand_wide.fillna(0.0)

        demand_wide.columns = [f"log_liters_{i}" for i in range(n)] # Rename the columns to log_liters_{i}
        obs_mask.columns    = [f"obs_mask_{i}"   for i in range(n)] # Rename the columns to obs_mask_{i}

        # Shared columns by (store, week): store-week features
        store_week_agg = (
            df.groupby(["store_code", "week_id"])[self._STORE_WEEK_COLS]
            .first()   # They are the same for all UPCs in that row
        )

        # Pivot table by UPC to the wide format. It is the same as above, but for the other columns in _PER_UPC_COLS.
        # Example
        # rows: (store_code, week_id, upc_code, lag_1_log_liters_sold ...)
        # to 
        # row (store_code=A, week_id=10) -> columns [upc_code=1, upc_code=2, upc_code=3, ...] with their lag_1_log_liters_sold ...
        per_upc_parts = []
        for long_col, short_col in zip(self._PER_UPC_COLS, self._PER_UPC_SHORT):
            part = self._pivot_upc_col(df, long_col, short_col, selected_upcs, n)
            if part is not None:
                per_upc_parts.append(part)
        for long_col, short_col in zip(self._PER_UPC_CAT_COLS, self._PER_UPC_CAT_SHORT):
            part = self._pivot_upc_col(df, long_col, short_col, selected_upcs, n,
                                        aggfunc="first", fill_value=0, as_int=True)
            if part is not None:
                per_upc_parts.append(part)

        # Final join of the pivot tables.
        wide = price_wide.join(demand_wide).join(obs_mask).join(store_week_agg)
        for part in per_upc_parts:
            wide = wide.join(part)

        return wide.reset_index()