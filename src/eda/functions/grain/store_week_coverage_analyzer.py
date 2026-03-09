import pandas as pd


class StoreWeekCoverageAnalyzer:
    """
    Analiza la cobertura store-week:
    - Cobertura global (store_week observados vs posibles)
    - Distribución de nº de weeks por store
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        store_col: str = "store_code",
        week_col: str = "week_id",
        min_weeks_for_good: int = 150,
    ) -> dict:
        """
        Args:
            df: DataFrame con grano (store, upc, week) sin duplicados.
            store_col: nombre de la columna de tienda.
            week_col: nombre de la columna de semana.
            min_weeks_for_good: umbral de weeks para marcar stores de baja cobertura.

        Returns:
            dict con:
                - n_stores: nº de stores distintos
                - n_weeks_global: nº de weeks distintas en el dataset
                - n_store_week_possible: combinaciones store-week posibles
                                         = n_stores × n_weeks_global
                - n_store_week_observed: combinaciones store-week observadas
                                         (pares únicos store-week)
                - store_week_density: densidad en [0,1]
                - store_week_density_pct: densidad en %
                - n_missing_store_weeks: nº de store-weeks ausentes
                - pct_missing_store_weeks: % de store-weeks ausentes
                - min_weeks_for_good: umbral usado
                - n_low_coverage_stores: nº de stores con weeks < umbral
                - low_coverage_stores: lista de IDs de esos stores
                - per_store_coverage: DataFrame con:
                    * store_col
                    * n_weeks_store
                    * coverage_pct
                    * is_low_coverage (bool)
        """
        for col in [store_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        # Stats globales
        n_stores = int(df[store_col].nunique(dropna=True))
        n_weeks_global = int(df[week_col].nunique(dropna=True))

        n_store_week_possible = n_stores * n_weeks_global

        # Pairs store-week observados (independientemente de UPCs)
        n_store_week_observed = int(
            df[[store_col, week_col]].drop_duplicates().shape[0]
        )

        if n_store_week_possible > 0:
            store_week_density = n_store_week_observed / n_store_week_possible
        else:
            store_week_density = 0.0

        store_week_density_pct = round(store_week_density * 100, 4)

        n_missing_store_weeks = max(n_store_week_possible - n_store_week_observed, 0)
        pct_missing_store_weeks = (
            round(n_missing_store_weeks / n_store_week_possible * 100, 4)
            if n_store_week_possible > 0
            else 0.0
        )

        # Distribución de weeks por store
        per_store = (
            df[[store_col, week_col]]
            .drop_duplicates()
            .groupby(store_col)[week_col]
            .nunique()
            .rename("n_weeks_store")
            .reset_index()
        )

        if n_weeks_global > 0:
            per_store["coverage_pct"] = (
                per_store["n_weeks_store"] / n_weeks_global * 100
            )
        else:
            per_store["coverage_pct"] = 0.0

        per_store["is_low_coverage"] = per_store["n_weeks_store"] < min_weeks_for_good

        low_cov = per_store[per_store["is_low_coverage"]]
        n_low_coverage_stores = int(low_cov.shape[0])
        low_coverage_stores = low_cov[store_col].tolist()

        return {
            "n_stores": n_stores,
            "n_weeks_global": n_weeks_global,
            "n_store_week_possible": n_store_week_possible,
            "n_store_week_observed": n_store_week_observed,
            "store_week_density": store_week_density,
            "store_week_density_pct": store_week_density_pct,
            "n_missing_store_weeks": n_missing_store_weeks,
            "pct_missing_store_weeks": pct_missing_store_weeks,
            "min_weeks_for_good": min_weeks_for_good,
            "n_low_coverage_stores": n_low_coverage_stores,
            "low_coverage_stores": low_coverage_stores,
            "per_store_coverage": per_store,
            "store_col": store_col,
            "week_col": week_col,
        }