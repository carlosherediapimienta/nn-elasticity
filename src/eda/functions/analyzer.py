import pandas as pd
from typing import Optional

class NanAnalyzer:
    """
    Analiza y reporta NaNs por columna.
    API pública: run().
    """

    def run(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """API pública. Retorna DataFrame con n_nans y pct_nans por columna."""
        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()
        counts = df[columns].isna().sum()
        pct = (counts / len(df) * 100).round(2)
        return pd.DataFrame({"n_nans": counts, "pct_nans": pct}, index=columns)



class CorrelationAnalyzer:
    """
    Calcula correlación de Spearman por grupo (vía rangos).
    API pública: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str = "store_code",
        min_obs: int = 30,
    ) -> pd.DataFrame:
        """API pública. Retorna DataFrame con group_col, corr y n por grupo."""
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
