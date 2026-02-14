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



class CorrelationAnalyzer:
    """
    Calculates Spearman correlation by group (via ranks).
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str = "store_code",
        min_obs: int = 30,
    ) -> pd.DataFrame:
        """Public API. Returns DataFrame with group_col, corr and n by group."""
        out = []
        mask = df[x_col].notna() & df[y_col].notna()
        for name, g in df.loc[mask].groupby(group_col):
            if len(g) < min_obs:
                continue
            x = g[x_col].astype(float).rank()
            y = g[y_col].astype(float).rank()
            r = x.corr(y)
            out.append({group_col: name, "corr": r, "n": len(g)})
        return pd.DataFrame(out)
