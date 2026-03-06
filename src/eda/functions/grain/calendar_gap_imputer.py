import numpy as np
import pandas as pd
from typing import Optional


class CalendarGapImputer:
    """
    Reconstructs the temporal axis for each (store, upc) series within its
    active span (first_week..last_week), inserting missing calendar weeks as
    new rows with NaN in time-varying columns and propagating static descriptive
    columns from existing rows of the same (store, upc).
    Public API: run().
    """

    _FLAG_COLS = [
        "is_imputed_calendar_row",
        "is_global_gap_week",
        "is_internal_gap",
        "gap_size_from_prev_obs",
        "weeks_since_last_obs",
    ]

    def run(
        self,
        df: pd.DataFrame,
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        week_col: str = "week_id",
        value_cols: Optional[list[str]] = None,
        global_gap_weeks: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Args:
            df: DataFrame original (grano: store x upc x week, sin duplicados).
            store_col: nombre columna de tienda.
            upc_col: nombre columna de producto.
            week_col: nombre columna de semana (int).
            value_cols: columnas de valor temporal que se ponen a NaN en filas
                        imputadas (ej. ["units_sold", "price", "promo"]).
                        El resto de columnas no-grano se consideran descriptivas
                        y se propagan desde una fila existente del mismo (store, upc).
                        Si None, TODAS las columnas no-grano se ponen a NaN
                        (comportamiento conservador).
            global_gap_weeks: semanas globalmente faltantes en el dataset.
                              Si None, se calculan automáticamente como las semanas
                              del rango [week_min, week_max] sin ninguna fila.

        Returns:
            DataFrame original + filas imputadas, ordenado por
            (store_col, upc_col, week_col), con columnas adicionales:
                - is_imputed_calendar_row  (int 0/1)
                - is_global_gap_week       (int 0/1)
                - is_internal_gap          (int 0/1)
                - gap_size_from_prev_obs   (int: nº de semanas del bloque de huecos)
                - weeks_since_last_obs     (int: posición 1-indexed dentro del bloque)
        """
        grain_cols = [store_col, upc_col, week_col]
        for col in grain_cols:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        # Determinar columnas de valor temporal y columnas estáticas descriptivas
        non_grain_cols = [
            c for c in df.columns
            if c not in grain_cols and c not in self._FLAG_COLS
        ]

        if value_cols is None:
            # Comportamiento conservador: todo va a NaN
            value_cols = non_grain_cols
            static_cols = []
        else:
            # Solo las value_cols van a NaN; el resto son descriptivas y se propagan
            value_cols = [c for c in value_cols if c in df.columns]
            static_cols = [c for c in non_grain_cols if c not in value_cols]

        # --- Global gap weeks ---
        observed_globally = set(int(w) for w in df[week_col].dropna().unique())
        week_min = min(observed_globally)
        week_max = max(observed_globally)
        if global_gap_weeks is None:
            global_gap_weeks_set = set(range(week_min, week_max + 1)) - observed_globally
        else:
            global_gap_weeks_set = set(int(w) for w in global_gap_weeks)

        # --- Marcar filas originales con flags a 0 ---
        df_out = df.copy()
        for flag in self._FLAG_COLS:
            df_out[flag] = 0

        # --- Span activo por (store, upc): [first_week, last_week] ---
        spans = (
            df.groupby([store_col, upc_col])[week_col]
            .agg(first_week="min", last_week="max")
            .reset_index()
        )
        spans["_weeks"] = spans.apply(
            lambda r: list(range(int(r["first_week"]), int(r["last_week"]) + 1)),
            axis=1,
        )
        expected = (
            spans[[store_col, upc_col, "_weeks"]]
            .explode("_weeks")
            .rename(columns={"_weeks": week_col})
            .reset_index(drop=True)
        )
        expected[week_col] = expected[week_col].astype(df[week_col].dtype)

        # --- Encontrar combinaciones (store, upc, week) faltantes ---
        existing = df[grain_cols].copy()
        existing["_exists"] = 1
        merged = expected.merge(existing, on=grain_cols, how="left")
        missing = (
            merged[merged["_exists"].isna()]
            .drop(columns=["_exists"])
            .copy()
        )

        if missing.empty:
            return df_out.sort_values(grain_cols).reset_index(drop=True)

        # --- Propagar columnas estáticas descriptivas desde (store, upc) ---
        if static_cols:
            static_lookup = (
                df.groupby([store_col, upc_col])[static_cols]
                .first()
                .reset_index()
            )
            missing = missing.merge(static_lookup, on=[store_col, upc_col], how="left")

        # --- Poner a NaN las columnas de valor temporal ---
        for col in value_cols:
            missing[col] = np.nan

        # --- Análisis de bloques de huecos (vectorizado) ---
        missing = missing.sort_values(grain_cols).reset_index(drop=True)

        missing["_prev_week"] = (
            missing.groupby([store_col, upc_col])[week_col].shift(1)
        )
        missing["_new_block"] = (
            (missing[week_col] - missing["_prev_week"] != 1)
            | missing["_prev_week"].isna()
        ).astype(int)
        missing["_block_id"] = (
            missing.groupby([store_col, upc_col])["_new_block"].cumsum()
        )

        missing["weeks_since_last_obs"] = (
            missing.groupby([store_col, upc_col, "_block_id"]).cumcount() + 1
        )
        block_sizes = (
            missing.groupby([store_col, upc_col, "_block_id"])
            .size()
            .rename("gap_size_from_prev_obs")
            .reset_index()
        )
        missing = missing.merge(
            block_sizes, on=[store_col, upc_col, "_block_id"], how="left"
        )
        missing = missing.drop(columns=["_prev_week", "_new_block", "_block_id"])

        # --- Asignar flags de tipo de hueco ---
        missing["is_imputed_calendar_row"] = 1
        missing["is_global_gap_week"] = (
            missing[week_col].isin(global_gap_weeks_set).astype(int)
        )
        missing["is_internal_gap"] = (
            (~missing[week_col].isin(global_gap_weeks_set)).astype(int)
        )

        # --- Combinar filas originales + imputadas ---
        df_final = pd.concat([df_out, missing], ignore_index=True)
        df_final = df_final.sort_values(grain_cols).reset_index(drop=True)
        return df_final