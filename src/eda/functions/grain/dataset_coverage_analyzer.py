import pandas as pd
from typing import Iterable, Optional


class DatasetCoverageAnalyzer:
    """
    Calcula dimensiones básicas del dataset y cobertura temporal (weeks)
    incluyendo semanas faltantes globales.
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        week_col: str = "week_id",
        expected_weeks: Optional[Iterable[int]] = None,
    ) -> dict:
        """
        Args:
            df: DataFrame con los datos.
            store_col: nombre de la columna de tienda.
            upc_col: nombre de la columna de producto/UPC.
            week_col: nombre de la columna temporal (semana).
            expected_weeks: iterable de IDs de semana esperados. Si es None,
                            se usa el rango completo [week_min, week_max].

        Returns:
            dict con, por ejemplo:
                - n_rows
                - n_stores
                - n_upcs
                - n_weeks_observed
                - week_min
                - week_max
                - n_expected_weeks
                - n_missing_weeks
                - pct_missing_weeks
                - weeks_missing (lista ordenada)
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        n_rows = int(len(df))
        n_stores = int(df[store_col].nunique(dropna=True))
        n_upcs = int(df[upc_col].nunique(dropna=True))

        week_series = df[week_col].dropna()
        if week_series.empty:
            raise ValueError(f"No hay valores no nulos en la columna de semanas '{week_col}'.")

        week_min = int(week_series.min())
        week_max = int(week_series.max())

        observed_weeks = sorted(set(int(w) for w in week_series.unique()))
        n_weeks_observed = len(observed_weeks)

        if expected_weeks is None:
            expected_weeks_set = set(range(week_min, week_max + 1))
        else:
            expected_weeks_set = set(int(w) for w in expected_weeks)

        observed_weeks_set = set(observed_weeks)
        missing_weeks = sorted(expected_weeks_set - observed_weeks_set)

        n_expected_weeks = len(expected_weeks_set)
        n_missing_weeks = len(missing_weeks)
        pct_missing_weeks = (
            round(n_missing_weeks / n_expected_weeks * 100, 4)
            if n_expected_weeks > 0
            else 0.0
        )

        return {
            "n_rows": n_rows,
            "n_stores": n_stores,
            "n_upcs": n_upcs,
            "n_weeks_observed": n_weeks_observed,
            "week_min": week_min,
            "week_max": week_max,
            "n_expected_weeks": n_expected_weeks,
            "n_missing_weeks": n_missing_weeks,
            "pct_missing_weeks": pct_missing_weeks,
            "weeks_missing": missing_weeks,
            "store_col": store_col,
            "upc_col": upc_col,
            "week_col": week_col,
        }