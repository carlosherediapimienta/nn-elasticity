import pandas as pd


class StoreUpcCoverageAnalyzer:
    """
    Analyzes the coverage by series (store, upc):
    - number of observed weeks (n_obs)
    - active temporal span (first_week..last_week)
    - coverage_ratio within the span
    - missing_within_span (missing weeks within the active span)
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
            week_col: week column (int).

        Returns:
            dict with:
                - per_series_coverage: DataFrame with columns
                    * store_col: store column
                    * upc_col: product/UPC column
                    * first_week: first week
                    * last_week: last week
                    * n_obs: number of observed weeks
                    * span_length_weeks: span length in weeks
                    * coverage_ratio: coverage ratio
                    * missing_within_span: missing weeks within the span
                - summary_stats: dict with summary statistics
                    * n_pairs: number of pairs
                    * n_obs: mean, p25, p50, p75, p90
                    * coverage_ratio: mean, p25, p50, p75, p90
                    * missing_within_span: mean, p25, p50, p75, p90
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # Grouping by series (store, upc)
        agg = (
            df.groupby([store_col, upc_col])[week_col]
            .agg(first_week="min", last_week="max", n_obs="nunique")
            .reset_index()
        )

        # Active span and coverage metrics
        agg["span_length_weeks"] = (
            agg["last_week"].astype(int) - agg["first_week"].astype(int) + 1
        )

        # Avoid rare divisions if span_length_weeks <= 0 (defensive)
        valid_span = agg["span_length_weeks"] > 0
        agg.loc[valid_span, "coverage_ratio"] = (
            agg.loc[valid_span, "n_obs"] / agg.loc[valid_span, "span_length_weeks"]
        )
        agg.loc[~valid_span, "coverage_ratio"] = pd.NA

        agg["missing_within_span"] = agg["span_length_weeks"] - agg["n_obs"]

        # Summaries
        n_pairs = int(agg.shape[0])

        def _q(series: pd.Series, q: float) -> float:
            return float(series.quantile(q)) if series.notna().sum() > 0 else float("nan")

        n_obs_series = agg["n_obs"]
        cov_series = agg["coverage_ratio"]
        miss_series = agg["missing_within_span"]

        summary_stats = {
            "n_pairs": n_pairs,
            "n_obs": {
                "mean": float(n_obs_series.mean()),
                "p25": _q(n_obs_series, 0.25),
                "p50": _q(n_obs_series, 0.50),
                "p75": _q(n_obs_series, 0.75),
                "p90": _q(n_obs_series, 0.90),
            },
            "coverage_ratio": {
                "mean": float(cov_series.mean()),
                "p25": _q(cov_series, 0.25),
                "p50": _q(cov_series, 0.50),
                "p75": _q(cov_series, 0.75),
                "p90": _q(cov_series, 0.90),
            },
            "missing_within_span": {
                "mean": float(miss_series.mean()),
                "p25": _q(miss_series, 0.25),
                "p50": _q(miss_series, 0.50),
                "p75": _q(miss_series, 0.75),
                "p90": _q(miss_series, 0.90),
            },
            "store_col": store_col,
            "upc_col": upc_col,
            "week_col": week_col,
        }

        return {
            "per_series_coverage": agg,
            "summary_stats": summary_stats,
        }