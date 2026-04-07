import pandas as pd


class PanelBalanceAnalyzer:
    """
    Analyzes if the panel is balanced and calculates its density
    relative to the hypothetical full panel (stores × upcs × weeks).
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
                - n_rows: filas reales en el dataset
                - n_stores: number of distinct stores
                - n_upcs: number of distinct UPCs
                - n_weeks: number of observed weeks (unique)
                - n_full_panel: size of the hypothetical full panel
                                = n_stores × n_upcs × n_weeks
                - density: density in [0,1]
                - density_pct: density in %
                - n_missing_cells: missing cells relative to the full panel
                - pct_missing_cells: % of missing cells
                - is_perfect_full_panel: True if the panel is completely full
        """
        for col in [store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

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