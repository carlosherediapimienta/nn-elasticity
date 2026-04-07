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
            df: DataFrame original (grain: store x upc x week, without duplicates).
            store_col: store column.
            upc_col: product column.
            week_col: week column (int).
            value_cols: temporal value columns that are set to NaN in imputed rows
                        imputed (for example ["units_sold", "price", "promo"]).
                        The rest of the non-grain columns are considered descriptive
                        and are propagated from an existing row of the same (store, upc).
                        If None, ALL non-grain columns are set to NaN
                        (conservative behavior).
            global_gap_weeks: global missing weeks in the dataset.
                              If None, they are calculated automatically as the weeks
                              in the range [week_min, week_max] without any rows.

        Returns:
            DataFrame original + imputed rows, ordered by
            (store_col, upc_col, week_col), with additional columns:
                - is_imputed_calendar_row  (int 0/1)
                - is_global_gap_week       (int 0/1)
                - is_internal_gap          (int 0/1)
                - gap_size_from_prev_obs   (int: number of weeks in the gap block)
                - weeks_since_last_obs     (int: 1-indexed position within the block)
        """
        grain_cols = [store_col, upc_col, week_col]
        for col in grain_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # Determine temporal value columns and descriptive static columns
        non_grain_cols = [
            c for c in df.columns
            if c not in grain_cols and c not in self._FLAG_COLS
        ]

        if value_cols is None:
            # Conservative behavior: everything goes to NaN
            value_cols = non_grain_cols
            static_cols = []
        else:
            # Only the value_cols go to NaN; the rest are descriptive and are propagated
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

        # --- Mark original rows with flags to 0 ---
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

        # --- Find missing combinations (store, upc, week) ---
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

        # --- Propagate descriptive static columns from (store, upc) ---
        if static_cols:
            static_lookup = (
                df.groupby([store_col, upc_col])[static_cols]
                .first()
                .reset_index()
            )
            missing = missing.merge(static_lookup, on=[store_col, upc_col], how="left")

        # --- Set the temporal value columns to NaN ---
        for col in value_cols:
            missing[col] = np.nan

        # --- Analysis of gap blocks (vectorized) ---
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

        # --- Assign gap type flags ---
        missing["is_imputed_calendar_row"] = 1
        missing["is_global_gap_week"] = (
            missing[week_col].isin(global_gap_weeks_set).astype(int)
        )
        missing["is_internal_gap"] = (
            (~missing[week_col].isin(global_gap_weeks_set)).astype(int)
        )

        # --- Combine original rows + imputed rows ---
        df_final = pd.concat([df_out, missing], ignore_index=True)
        df_final = df_final.sort_values(grain_cols).reset_index(drop=True)
        return df_final