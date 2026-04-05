import pandas as pd


class StoreWeekWidthAnalyzer:
    """
    Analyzes the width of the panel by store-week:
    number of distinct UPCs by (store, week).
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        week_col: str = "week_id",
    ) -> dict:
        """
        Args:
            df: DataFrame with grain (store, upc, week) without duplicates.
            store_col: store column.
            upc_col: product/UPC column.
            week_col: week column.

        Returns:
            dict with:
                - per_store_week_width: DataFrame with columns
                    * store_col: store column
                    * week_col: week column
                    * n_upcs_store_week: number of distinct UPCs per store-week
                - summary_stats: dict with summary statistics
                    * count, mean, std, min, max
                    * p05, p25, p50, p75, p95
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # number of distinct UPCs per store-week
        width_series = (
            df.groupby([store_col, week_col])[upc_col]
            .nunique()
            .rename("n_upcs_store_week")
        )

        per_store_week_width = width_series.reset_index()

        # Basic statistics
        s = per_store_week_width["n_upcs_store_week"]
        summary_stats = {
            "count": int(s.shape[0]),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": int(s.min()) if s.shape[0] > 0 else 0,
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": int(s.max()) if s.shape[0] > 0 else 0,
        }

        return {
            "per_store_week_width": per_store_week_width,
            "summary_stats": summary_stats,
            "store_col": store_col,
            "upc_col": upc_col,
            "week_col": week_col,
        }