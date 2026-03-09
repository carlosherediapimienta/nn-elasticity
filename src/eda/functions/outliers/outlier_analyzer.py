import pandas as pd
from typing import Optional, Union


class OutlierAnalyzer:
    """
    Detects outliers using ±3 std method or IQR method.
    Supports global (whole column) or by-group (e.g. per UPC) bounds.
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: str = "std",
        n_std: float = 3.0,
        group_col: Optional[Union[str, list[str]]] = None,
        return_flagged_df: bool = False,
    ) -> Union[pd.DataFrame, dict]:
        """
        Public API. Returns outlier statistics and optionally the dataframe with
        outlier flags for filtering (e.g. for OLS sanity checks).

        Args:
            df: DataFrame with data.
            columns: list of column names to analyze.
            method: 'std' (±n std) or 'iqr' (interquartile range 1.5×).
            n_std: number of std deviations for outlier threshold (default 3).
            group_col: if set (e.g. 'upc_code' or ['store_code','upc_code']),
                       bounds are computed per group; otherwise global.
            return_flagged_df: if True, return dict with 'summary' and 'flagged_df'
                               (df plus is_outlier_<col> columns). Otherwise
                               return only the summary DataFrame.

        Returns:
            If return_flagged_df is False: DataFrame with columns
                variable, n_outliers, pct_outliers, lower_bound, upper_bound, method, scope.
            If return_flagged_df is True: dict with
                'summary': same DataFrame,
                'flagged_df': df with extra columns is_outlier_<col> (bool).
        """
        if group_col is not None:
            if isinstance(group_col, str):
                group_col = [group_col]
            for c in group_col:
                if c not in df.columns:
                    raise ValueError(f"group_col '{c}' not in DataFrame.")
        scope = "global" if group_col is None else f"by_{'_'.join(group_col)}"

        results = []
        flagged = df.copy()

        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not in DataFrame.")

            if group_col is None:
                summary_row, is_outlier = self._outliers_global(df[col], method, n_std)
            else:
                summary_row, is_outlier = self._outliers_by_group(
                    df, col, group_col, method, n_std
                )

            summary_row["variable"] = col
            summary_row["scope"] = scope
            results.append(summary_row)
            flagged[f"is_outlier_{col}"] = is_outlier.reindex(df.index).fillna(False).values

        summary_df = pd.DataFrame(results)

        if return_flagged_df:
            return {"summary": summary_df, "flagged_df": flagged}
        return summary_df

    def _outliers_global(
        self, series: pd.Series, method: str, n_std: float
    ) -> tuple[dict, pd.Series]:
        """Compute global bounds and outlier mask. Returns (summary_dict, boolean series)."""
        s = series.dropna()
        if method == "std":
            mean = s.mean()
            std = s.std()
            if pd.isna(std) or std == 0:
                lower, upper = mean, mean
                method_desc = f"±{n_std} std"
            else:
                lower = mean - n_std * std
                upper = mean + n_std * std
                method_desc = f"±{n_std} std"
        elif method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            method_desc = "IQR (1.5)"
        else:
            raise ValueError("method must be 'std' or 'iqr'")

        outliers = (series < lower) | (series > upper)
        n_outliers = int(outliers.sum())
        n_valid = int(series.notna().sum())
        pct_outliers = round(n_outliers / n_valid * 100, 2) if n_valid > 0 else 0.0

        summary = {
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "method": method_desc,
        }
        return summary, outliers

    def _outliers_by_group(
        self,
        df: pd.DataFrame,
        col: str,
        group_col: list[str],
        method: str,
        n_std: float,
    ) -> tuple[dict, pd.Series]:
        """Compute per-group bounds via transform (no merge needed)."""
        if method == "iqr":
            q1 = df.groupby(group_col)[col].transform(lambda x: x.quantile(0.25))
            q3 = df.groupby(group_col)[col].transform(lambda x: x.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            method_desc = "IQR (1.5)"
        elif method == "std":
            mean = df.groupby(group_col)[col].transform("mean")
            std = df.groupby(group_col)[col].transform("std").fillna(0)
            lower = mean - n_std * std
            upper = mean + n_std * std
            method_desc = f"±{n_std} std"
        else:
            raise ValueError("method must be 'std' or 'iqr'")

        outliers = (df[col] < lower) | (df[col] > upper)
        outliers = outliers.fillna(False)

        n_outliers = int(outliers.sum())
        n_valid = int(df[col].notna().sum())
        pct_outliers = round(n_outliers / n_valid * 100, 2) if n_valid > 0 else 0.0

        summary = {
            "n_outliers": n_outliers,
            "pct_outliers": pct_outliers,
            "lower_bound": None,
            "upper_bound": None,
            "method": method_desc,
        }
        return summary, outliers