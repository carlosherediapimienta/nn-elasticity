import pandas as pd


class PanelBalanceAnalyzer:
    """
    Analiza si el panel está balanceado y calcula su densidad
    respecto al panel completo hipotético (stores × upcs × weeks).
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
            dict con, por ejemplo:
                - n_rows: filas reales en el dataset
                - n_stores: nº de stores distintos
                - n_upcs: nº de UPCs distintos
                - n_weeks: nº de weeks observadas (únicas)
                - n_full_panel: tamaño del panel completo hipotético
                                = n_stores × n_upcs × n_weeks
                - density: densidad en [0,1]
                - density_pct: densidad en %
                - n_missing_cells: celdas ausentes respecto al panel completo
                - pct_missing_cells: % de celdas ausentes
                - is_perfect_full_panel: True si el panel está totalmente lleno
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        n_rows = int(len(df))
        n_stores = int(df[store_col].nunique(dropna=True))
        n_upcs = int(df[upc_col].nunique(dropna=True))
        n_weeks = int(df[week_col].nunique(dropna=True))

        n_full_panel = n_stores * n_upcs * n_weeks

        if n_full_panel > 0:
            density = n_rows / n_full_panel
        else:
            density = 0.0

        density_pct = round(density * 100, 4)

        n_missing_cells = max(n_full_panel - n_rows, 0)
        pct_missing_cells = (
            round(n_missing_cells / n_full_panel * 100, 4)
            if n_full_panel > 0
            else 0.0
        )

        is_perfect_full_panel = (n_full_panel > 0) and (n_rows == n_full_panel)

        return {
            "n_rows": n_rows,
            "n_stores": n_stores,
            "n_upcs": n_upcs,
            "n_weeks": n_weeks,
            "n_full_panel": n_full_panel,
            "density": density,
            "density_pct": density_pct,
            "n_missing_cells": n_missing_cells,
            "pct_missing_cells": pct_missing_cells,
            "is_perfect_full_panel": is_perfect_full_panel,
            "store_col": store_col,
            "upc_col": upc_col,
            "week_col": week_col,
        }