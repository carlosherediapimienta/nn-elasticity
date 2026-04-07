import numpy as np
import pandas as pd


class BootstrapSummarizer:
    """
    Summarizes bootstrap results into mean, std and empirical 95% CIs.
    Two aggregation levels are provided:
    - ``summarize_raw``: collapses per-run directional results
    (upc_i → upc_j) across bootstrap runs.
    - ``summarize_cross``: collapses per-run symmetrized cross-elasticities
    (canonical pair upc_a ↔ upc_b) across bootstrap runs.
    CIs are empirical percentiles (p2.5 - p97.5).
    """
    # Function to calculate the 2.5th percentile
    @staticmethod
    def _q025(x: pd.Series) -> float:
        return np.percentile(x, 2.5)

    # Function to calculate the 97.5th percentile
    @staticmethod
    def _q975(x: pd.Series) -> float:
        return np.percentile(x, 97.5)

    # Method to summarize the raw bootstrap results
    def summarize_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the bootstrap results per (store, pair, upc_i, upc_j)."""
        return (
            df
            .groupby(["store_code", "pair_id", "upc_i", "upc_j"])
            .agg(
                own_elasticity_mean=("own_elasticity", "mean"),
                own_elasticity_std=("own_elasticity", "std"),
                own_elasticity_ci_low=("own_elasticity", self._q025),
                own_elasticity_ci_high=("own_elasticity", self._q975),
                cross_elasticity_mean=("cross_elasticity", "mean"),
                cross_elasticity_std=("cross_elasticity", "std"),
                cross_elasticity_ci_low=("cross_elasticity", self._q025),
                cross_elasticity_ci_high=("cross_elasticity", self._q975),
                mae_val_mean=("mae_val", "mean"),
                mae_val_std=("mae_val", "std"),
                rmse_val_mean=("rmse_val", "mean"),
                rmse_val_std=("rmse_val", "std"),
                r2_val_mean=("r2_val", "mean"),
                r2_val_std=("r2_val", "std"),
            )
            .reset_index()
        )

    def summarize_cross(self, sym_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate symmetrized cross-elasticities across bootstrap runs."""
        return (
            sym_df
            .groupby(["store_code", "pair_id", "upc_a", "upc_b"])
            .agg(
                cross_elasticity_sym_mean=("cross_elasticity_sym", "mean"),
                cross_elasticity_sym_std=("cross_elasticity_sym", "std"),
                cross_elasticity_sym_ci_low=("cross_elasticity_sym", self._q025),
                cross_elasticity_sym_ci_high=("cross_elasticity_sym", self._q975),
                mae_val_mean=("mae_val_mean", "mean"),
                mae_val_std=("mae_val_mean", "std"),
                rmse_val_mean=("rmse_val_mean", "mean"),
                rmse_val_std=("rmse_val_mean", "std"),
                r2_val_mean=("r2_val_mean", "mean"),
                r2_val_std=("r2_val_mean", "std"),
            )
            .reset_index()
        )