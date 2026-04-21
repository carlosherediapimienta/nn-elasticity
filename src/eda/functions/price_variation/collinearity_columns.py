import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


class CollinearityAnalyzer:
    """
    Analyzes multicollinearity among numeric features:
    - Pairwise correlation matrix
    - High-correlation feature pairs
    - Variance Inflation Factor (VIF) per feature

    Public API: run().

    By default, run() returns a compact, presentation-ready output:
    summary + top tables + focused correlation matrix.
    """

    def run(
        self,
        df: pd.DataFrame,
        columns: list[str],
        corr_method: str = "spearman",
        corr_threshold: float = 0.8,
        vif_threshold: float = 5.0,
        drop_na: bool = True,
        top_corr: int = 20,
        top_vif: int = 20,
        return_full: bool = False,
    ) -> dict:
        """
        Args:
            df: Input DataFrame containing candidate features.
            columns: Numeric columns to include in the collinearity study.
            corr_method: Correlation method ('pearson', 'spearman', 'kendall').
            corr_threshold: Absolute-correlation threshold for flagging feature pairs.
            vif_threshold: VIF threshold for flagging multicollinearity risk.
            drop_na: If True, drops rows with NaN after numeric coercion.
            top_corr: Number of top high-correlation pairs to keep in compact output.
            top_vif: Number of top VIF rows to keep in compact output.
            return_full: If True, also returns full corr_matrix/high_corr_pairs/vif_table.

        Returns:
            dict with compact outputs:
                - summary
                - high_corr_pairs_top
                - vif_table_top
                - focus_corr_matrix

            If return_full=True, also includes:
                - corr_matrix
                - high_corr_pairs
                - vif_table
        """
        if not columns:
            raise ValueError("`columns` cannot be empty.")

        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        work = df[columns].copy()

        # Force numeric dtype to avoid mixed-type issues.
        for col in columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        n_rows_input = int(len(work))

        if drop_na:
            work = work.dropna()

        n_rows_used = int(len(work))
        if n_rows_used == 0:
            raise ValueError("No valid rows remain after numeric coercion and NA handling.")

        # Correlation matrix.
        corr_matrix = work.corr(method=corr_method)

        # Extract upper-triangle high-correlation pairs.
        high_corr_rows = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                var_a = cols[i]
                var_b = cols[j]
                corr_val = corr_matrix.loc[var_a, var_b]
                if pd.notna(corr_val) and abs(corr_val) >= corr_threshold:
                    high_corr_rows.append(
                        {
                            "var_a": var_a,
                            "var_b": var_b,
                            "corr": float(corr_val),
                            "abs_corr": float(abs(corr_val)),
                        }
                    )

        if high_corr_rows:
            high_corr_pairs = pd.DataFrame(high_corr_rows).sort_values(
                by="abs_corr", ascending=False
            )
        else:
            high_corr_pairs = pd.DataFrame(
                columns=["var_a", "var_b", "corr", "abs_corr"]
            )

        # Prepare design matrix for VIF (drop constant features).
        X = work.loc[:, work.nunique(dropna=False) > 1].copy()

        vif_rows = []
        if X.shape[1] >= 2:
            x_values = X.values.astype(float)
            for idx, col in enumerate(X.columns):
                vif_val = variance_inflation_factor(x_values, idx)
                vif_rows.append(
                    {
                        "variable": col,
                        "vif": float(vif_val),
                        "flag_high_vif": bool(vif_val >= vif_threshold),
                    }
                )

        if vif_rows:
            vif_table = pd.DataFrame(vif_rows).sort_values(by="vif", ascending=False)
        else:
            vif_table = pd.DataFrame(columns=["variable", "vif", "flag_high_vif"])

        summary = {
            "n_rows_input": n_rows_input,
            "n_rows_used": n_rows_used,
            "n_features_input": int(len(columns)),
            "n_features_used_for_vif": int(X.shape[1]),
            "corr_method": corr_method,
            "corr_threshold": float(corr_threshold),
            "vif_threshold": float(vif_threshold),
            "n_high_corr_pairs": int(len(high_corr_pairs)),
            "n_high_vif_features": int(vif_table["flag_high_vif"].sum())
            if not vif_table.empty
            else 0,
        }

        # Compact outputs (for easy visualization).
        high_corr_pairs_top = high_corr_pairs.head(top_corr).copy()
        vif_table_top = vif_table.head(top_vif).copy()

        focus_vars = set()
        if not high_corr_pairs_top.empty:
            focus_vars.update(high_corr_pairs_top["var_a"].tolist())
            focus_vars.update(high_corr_pairs_top["var_b"].tolist())

        if not vif_table.empty:
            focus_vars.update(
                vif_table.loc[vif_table["flag_high_vif"], "variable"].head(top_vif).tolist()
            )

        focus_vars = sorted(focus_vars)
        if len(focus_vars) >= 2:
            focus_corr_matrix = corr_matrix.loc[focus_vars, focus_vars]
        else:
            focus_corr_matrix = pd.DataFrame()

        result = {
            "summary": summary,
            "high_corr_pairs_top": high_corr_pairs_top,
            "vif_table_top": vif_table_top,
            "focus_corr_matrix": focus_corr_matrix,
        }

        if return_full:
            result["corr_matrix"] = corr_matrix
            result["high_corr_pairs"] = high_corr_pairs
            result["vif_table"] = vif_table

        return result