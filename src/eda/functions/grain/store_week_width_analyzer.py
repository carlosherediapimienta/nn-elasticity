import pandas as pd


class StoreWeekWidthAnalyzer:
    """
    Analiza el "ancho" del panel por tienda-semana:
    nº de UPCs distintos por (store, week).
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
            week_col: nombre de la columna de semana.

        Returns:
            dict con:
                - per_store_week_width: DataFrame con columnas
                    * store_col
                    * week_col
                    * n_upcs_store_week
                - summary_stats: dict con resumen estadístico
                    * count, mean, std, min, max
                    * p05, p25, p50, p75, p95
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        # nº de UPCs distintos por tienda-semana
        width_series = (
            df.groupby([store_col, week_col])[upc_col]
            .nunique()
            .rename("n_upcs_store_week")
        )

        per_store_week_width = width_series.reset_index()

        # Estadísticos básicos
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