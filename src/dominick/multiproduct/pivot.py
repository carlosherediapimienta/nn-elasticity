import pandas as pd

class MultiProductPivoter:
    """
    Pivots from long format to wide format with obs_mask columns.

    For each product i:
      - log_price_i:   last-known price (forward-filled) if absent, else observed.
      - log_liters_i:  observed demand, NaN if absent.
      - obs_mask_i:    1 if demand was observed, 0 otherwise.
      - lag_1_i, lag_2_i, lag_4_i:  precomputed demand lags.
      - roll_4_i, roll_8_i, roll_13_i: precomputed rolling means.
      - miss_lag_1_i ... miss_roll_13_i: missing indicators.
      - weeks_since_seen_upc_i, weeks_since_seen_store_upc_i, liters_per_upc_i.

    Shared per (store, week):
      - on_promo, promo_intensity_store_week
      - week_rank, sin_52, cos_52, sin_26, cos_26, sin_13, cos_13

    Public API: run(df, selected_upcs) → wide df
    """

    # columnas que son iguales para todos los UPCs en un (store, week)
    _STORE_WEEK_COLS = [
        "on_promo", "promo_intensity_store_week",
        "week_rank",
        "sin_52", "cos_52", "sin_26", "cos_26", "sin_13", "cos_13",
    ]

    # columnas por UPC que se pivotarán → nombre_i
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
    ]

    # nombres cortos que tendrán en el wide → col_{i}
    _PER_UPC_SHORT = [
        "lag_1", "lag_2", "lag_4",
        "roll_4", "roll_8", "roll_13",
        "miss_lag_1", "miss_lag_2", "miss_lag_4",
        "miss_roll_4", "miss_roll_8", "miss_roll_13",
        "weeks_seen_upc", "weeks_seen_store_upc", "liters_per_upc",
    ]

    def run(self, df: pd.DataFrame, selected_upcs: list) -> pd.DataFrame:
        df = df[df["upc_code"].isin(selected_upcs)].copy()
        n = len(selected_upcs)

        # ── precio ────────────────────────────────────────────────────────
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
        price_wide = (
            price_wide
            .groupby("store_code", group_keys=False)
            .apply(lambda g: g.ffill().bfill())
        )
        for col in price_wide.columns:
            price_wide[col] = price_wide[col].fillna(price_wide[col].mean())
        price_wide.columns = [f"log_price_{i}" for i in range(n)]

        # ── demanda + obs_mask ────────────────────────────────────────────
        demand_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_liters_sold",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)

        obs_mask = demand_wide.notna().astype(float)
        demand_wide = demand_wide.fillna(0.0)

        demand_wide.columns = [f"log_liters_{i}" for i in range(n)]
        obs_mask.columns    = [f"obs_mask_{i}"   for i in range(n)]

        # ── columnas compartidas por (store, week) ────────────────────────
        store_week_agg = (
            df.groupby(["store_code", "week_id"])[self._STORE_WEEK_COLS]
            .first()   # son iguales para todos los UPCs de esa fila
        )

        # ── columnas por UPC → wide ───────────────────────────────────────
        per_upc_parts = []
        for long_col, short_col in zip(self._PER_UPC_COLS, self._PER_UPC_SHORT):
            if long_col not in df.columns:
                continue
            pivoted = df.pivot_table(
                index=["store_code", "week_id"],
                columns="upc_code",
                values=long_col,
                aggfunc="mean",
            ).reindex(columns=selected_upcs)
            pivoted = pivoted.fillna(0.0)
            pivoted.columns = [f"{short_col}_{i}" for i in range(n)]
            per_upc_parts.append(pivoted)

        # ── join final ────────────────────────────────────────────────────
        wide = price_wide.join(demand_wide).join(obs_mask).join(store_week_agg)
        for part in per_upc_parts:
            wide = wide.join(part)

        return wide.reset_index()