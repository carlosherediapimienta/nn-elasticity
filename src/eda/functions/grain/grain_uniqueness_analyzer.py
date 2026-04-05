import pandas as pd


class GrainUniquenessAnalyzer:
    """
    Validates that a set of columns defines the grain (uniqueness) of the DataFrame.
    Public API: run().
    """

    def run(self, df: pd.DataFrame, grain_cols: list[str]) -> pd.DataFrame:
        """
        Args:
            df: DataFrame with the data.
            grain_cols: columns that define the grain, for example
                        ["week_id", "store_code", "upc_code"].

        Returns:
            DataFrame with a single row with:
                - grain_cols: grain columns (as a string separated by commas)
                - n_rows: total number of rows
                - n_unique_keys: number of unique grain combinations
                - n_duplicate_keys: number of grain keys with more than 1 row
                - pct_duplicate_keys: % of duplicate grain keys
                - n_rows_with_missing_in_grain: number of rows with at least one NaN in the grain
                - is_unique_grain: bool, True if the grain is perfectly unique and without NaNs
        """
        missing_cols = [c for c in grain_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing grain columns in the DataFrame: {missing_cols}")

        grain_df = df[grain_cols]

        # Rows with any NaN in the grain
        n_rows_with_missing = int(grain_df.isna().any(axis=1).sum())

        n_rows = int(len(df))

        # Count of grain combinations (including NaNs if any)
        key_counts = (
            grain_df
            .groupby(grain_cols, dropna=False)
            .size()
        )

        n_unique_keys = int(key_counts.shape[0])
        n_duplicate_keys = int((key_counts > 1).sum())

        if n_unique_keys > 0:
            pct_duplicate_keys = round(n_duplicate_keys / n_unique_keys * 100, 4)
        else:
            pct_duplicate_keys = 0.0

        is_unique_grain = (n_rows == n_unique_keys) and (n_rows_with_missing == 0)

        result = pd.DataFrame(
            [{
                "grain_cols": ", ".join(grain_cols),
                "n_rows": n_rows,
                "n_unique_keys": n_unique_keys,
                "n_duplicate_keys": n_duplicate_keys,
                "pct_duplicate_keys": pct_duplicate_keys,
                "n_rows_with_missing_in_grain": n_rows_with_missing,
                "is_unique_grain": is_unique_grain,
            }]
        )

        return result