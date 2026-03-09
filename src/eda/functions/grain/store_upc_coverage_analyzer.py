import pandas as pd


class StoreUpcCoverageAnalyzer:
    """
    Analiza la cobertura por serie (store, upc):
    - nº de semanas observadas (n_obs)
    - span temporal activo (first_week..last_week)
    - coverage_ratio dentro del span
    - missing_within_span (huecos dentro del rango activo)
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
            df: DataFrame con grano (store, upc, week) sin duplicados.
            store_col: nombre de la columna de tienda.
            upc_col: nombre de la columna de producto/UPC.
            week_col: nombre de la columna de semana (int).

        Returns:
            dict con:
                - per_series_coverage: DataFrame por (store, upc) con columnas
                    * store_col
                    * upc_col
                    * first_week
                    * last_week
                    * n_obs                (nº de weeks observadas)
                    * span_length_weeks    (last_week - first_week + 1)
                    * coverage_ratio       (n_obs / span_length_weeks)
                    * missing_within_span  (span_length_weeks - n_obs)
                - summary_stats: dict con resúmenes de:
                    * n_pairs
                    * n_obs: mean, p25, p50, p75, p90
                    * coverage_ratio: mean, p25, p50, p75, p90
                    * missing_within_span: mean, p25, p50, p75, p90
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        # Agrupación por serie (store, upc)
        agg = (
            df.groupby([store_col, upc_col])[week_col]
            .agg(first_week="min", last_week="max", n_obs="nunique")
            .reset_index()
        )

        # Span activo y métricas de cobertura
        agg["span_length_weeks"] = (
            agg["last_week"].astype(int) - agg["first_week"].astype(int) + 1
        )

        # Evitar divisiones raras si span_length_weeks <= 0 (defensivo)
        valid_span = agg["span_length_weeks"] > 0
        agg.loc[valid_span, "coverage_ratio"] = (
            agg.loc[valid_span, "n_obs"] / agg.loc[valid_span, "span_length_weeks"]
        )
        agg.loc[~valid_span, "coverage_ratio"] = pd.NA

        agg["missing_within_span"] = agg["span_length_weeks"] - agg["n_obs"]

        # Resúmenes
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