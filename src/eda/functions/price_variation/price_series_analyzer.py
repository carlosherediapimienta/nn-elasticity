import pandas as pd


class PriceVariationAnalyzer:
    """
    Analyzes the price variation within each series (store, upc) to
    evaluate the practical identifiability of elasticity (own).
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        price_col: str = "log_price_per_liter",
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        week_col: str = "week_id",
        price_round_decimals: int = 3,
    ) -> dict:
        """
        Args:
            df: DataFrame with grain (store, upc, week) and price column.
            price_col: price column in log (e.g. log_price_per_liter).
            store_col, upc_col, week_col: grain columns.
            price_round_decimals: number of decimals to round when counting price levels.

        Returns:
            dict with per_series (DataFrame by store, upc) and summary_stats
            (n_pairs, n_constant_price_series, pct_constant_price_series,
             quantiles of n_price_levels, n_price_changes, price_range_log).
        """
        for col in [price_col, store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        df_work = df[[store_col, upc_col, week_col, price_col]].copy()
        df_work = df_work.dropna(subset=[price_col, week_col])
        df_work["_price_rounded"] = df_work[price_col].round(price_round_decimals)

        agg_list = []
        for (s, u), g in df_work.groupby([store_col, upc_col]):
            g = g.sort_values(week_col)
            n_obs = len(g)
            n_price_levels = int(g["_price_rounded"].nunique())
            diffs = g[price_col].diff()
            n_price_changes = int((diffs.abs() > 1e-12).sum())
            price_range_log = float(g[price_col].max() - g[price_col].min())
            agg_list.append({
                store_col: s,
                upc_col: u,
                "n_obs": n_obs,
                "n_price_levels": n_price_levels,
                "n_price_changes": n_price_changes,
                "price_range_log": price_range_log,
                "is_constant_price": n_price_changes == 0,
            })
        per_series = pd.DataFrame(agg_list)

        n_pairs = int(per_series.shape[0])
        n_constant_price_series = int(per_series["is_constant_price"].sum())
        pct_constant_price_series = (
            round(n_constant_price_series / n_pairs * 100, 4) if n_pairs > 0 else 0.0
        )

        def _quantiles(s: pd.Series) -> dict:
            q = s.dropna()
            if q.shape[0] == 0:
                return {"mean": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p90": float("nan")}
            return {
                "mean": float(q.mean()),
                "p25": float(q.quantile(0.25)),
                "p50": float(q.quantile(0.50)),
                "p75": float(q.quantile(0.75)),
                "p90": float(q.quantile(0.90)),
            }

        summary_stats = {
            "n_pairs": n_pairs,
            "n_constant_price_series": n_constant_price_series,
            "pct_constant_price_series": pct_constant_price_series,
            "n_price_levels": _quantiles(per_series["n_price_levels"]),
            "n_price_changes": _quantiles(per_series["n_price_changes"]),
            "price_range_log": _quantiles(per_series["price_range_log"]),
            "price_col": price_col,
            "store_col": store_col,
            "upc_col": upc_col,
            "week_col": week_col,
        }

        return {
            "per_series": per_series,
            "summary_stats": summary_stats,
        }