import pandas as pd


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