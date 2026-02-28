import pandas as pd

class MultiProductPivoter:
    """
    Pivots from long format to wide format with obs_mask columns.

    For each product i:
      - log_price_i:   last-known price (forward-filled) if absent, else observed.
      - log_liters_i:  observed demand, NaN if absent.
      - obs_mask_i:    1 if demand was observed, 0 otherwise.

    Public API: run(df, selected_upcs) → wide df
    """

    def run(self, df: pd.DataFrame, selected_upcs: list) -> pd.DataFrame:
        df = df[df["upc_code"].isin(selected_upcs)].copy()

        price_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_price_per_liter",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)

        demand_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_liters_sold",
            aggfunc="mean",
        ).reindex(columns=selected_upcs)

        # obs_mask: 1 where demand was observed, 0 where NaN
        obs_mask = demand_wide.notna().astype(float)

        # Forward-fill missing prices within each store, then fill remaining with column mean
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

        # Fill missing demand with 0 (won't enter loss thanks to obs_mask)
        demand_wide = demand_wide.fillna(0.0)

        # Rename columns
        n = len(selected_upcs)
        price_wide.columns  = [f"log_price_{i}"  for i in range(n)]
        demand_wide.columns = [f"log_liters_{i}" for i in range(n)]
        obs_mask.columns    = [f"obs_mask_{i}"   for i in range(n)]

        promo_agg = (
            df.groupby(["store_code", "week_id"])[
                ["on_promo", "promo_B", "promo_C", "promo_S"]
            ].max()
        )

        wide = price_wide.join(demand_wide).join(obs_mask).join(promo_agg)
        return wide.reset_index()