import pandas as pd
from typing import Optional


class NanAnalyzer:
    """
    Analyzes and reports NaNs by column.
    Public API: run().
    """

    def run(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Public API. Returns DataFrame with n_nans and pct_nans by column."""
        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()
        counts = df[columns].isna().sum()
        pct = (counts / len(df) * 100).round(2)
        return pd.DataFrame({"n_nans": counts, "pct_nans": pct}, index=columns)