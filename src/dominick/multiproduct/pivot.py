import pandas as pd

class MultiProductPivoter:
    """
    Pivots from long format (1 row per product) to wide format (n products per row).
    Generates columns log_price_0..n-1 and log_liters_0..n-1.
    Public API: run(df, selected_upcs) → wide df
    """
    def run(self, df: pd.DataFrame, selected_upcs: list) -> pd.DataFrame:
        df = df[df["upc_code"].isin(selected_upcs)].copy()

        price_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_price_per_liter"
        )[selected_upcs]
        price_wide.columns = [f"log_price_{i}" for i in range(len(selected_upcs))]

        demand_wide = df.pivot_table(
            index=["store_code", "week_id"],
            columns="upc_code",
            values="log_liters_sold"
        )[selected_upcs]
        demand_wide.columns = [f"log_liters_{i}" for i in range(len(selected_upcs))]

        promo_agg = df.groupby(["store_code", "week_id"])[
            ["on_promo", "promo_B", "promo_C", "promo_S"]
        ].max()

        wide = price_wide.join(demand_wide).join(promo_agg)
        return wide.reset_index()