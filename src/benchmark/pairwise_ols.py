import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import BenchmarkConfig

class PairwiseElasticityPipeline:
    """
    Log-log OLS benchmark for own- and cross-price elasticities.
    For each (store, pair, upc_i, upc_j) group present in both train and val,
    fits the model:
        log_v_i ~ log_p_i + log_p_j + [control_cols]
    where ``log_p_i`` captures own-price elasticity and ``log_p_j`` cross-price
    elasticity. Groups with insufficient price variation or too few observations
    are silently skipped. Fitting errors are caught and returned with
    ``status="error"`` so a single failure never aborts the full run.
    """
    # Group keys to use for grouping the data
    GROUP_KEYS = ["store_code", "pair_id", "upc_i", "upc_j"]

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._formula = self._build_formula()

    def _build_formula(self) -> str:
        # Build the right-hand side of the formula
        rhs = "log_p_i + log_p_j" # Own-price elasticity + Cross-price elasticity
        if self.config.control_cols:
            rhs += " + " + " + ".join(self.config.control_cols) # Add the control columns
        # Return the formula: log_v_i ~ log_p_i + log_p_j + [control_cols]
        return f"log_v_i ~ {rhs}"

    def _is_valid(self, g_train: pd.DataFrame, g_val: pd.DataFrame) -> bool:
        # Check if the training data has enough observations
        if len(g_train) < self.config.min_obs:
            return False
        # Check if the validation data has any observations
        if len(g_val) == 0:
            return False
        # Check if the training data has enough unique prices
        if g_train["log_p_i"].nunique() < 2:
            return False
        # Check if the training data has enough unique cross-prices
        if g_train["log_p_j"].nunique() < 2:
            return False
        return True

    def _fit_single(self, key: tuple, g_train: pd.DataFrame, g_val: pd.DataFrame) -> dict | None:
        store_code, pair_id, upc_i, upc_j = key
        # Try to fit the model
        try:
            # Fit the model (OLS with robust covariance)
            fit = smf.ols(formula=self._formula, data=g_train).fit(
                cov_type=self.config.robust_cov_type,
            )
            # Predict the validation data
            pred_val = fit.predict(g_val)
            # Get the actual values of the validation data
            y_val = g_val["log_v_i"]

            # Calculate the residuals
            residuals = y_val - pred_val
            # Calculate the MAE
            mae_val = np.mean(np.abs(residuals))
            # Calculate the RMSE
            rmse_val = np.sqrt(np.mean(residuals ** 2))
            # Calculate the R²
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            r2_val = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

            # Get the confidence intervals
            ci = fit.conf_int()

            # Return the results
            return {
                "store_code": store_code,
                "pair_id": pair_id,
                "upc_i": upc_i,
                "upc_j": upc_j,
                "status": "ok",
                "n_train": len(g_train),
                "n_val": len(g_val),
                "own_elasticity": fit.params.get("log_p_i", np.nan),
                "own_elasticity_ci_low": ci.loc["log_p_i", 0] if "log_p_i" in ci.index else np.nan,
                "own_elasticity_ci_high": ci.loc["log_p_i", 1] if "log_p_i" in ci.index else np.nan,
                "own_elasticity_p_value": fit.pvalues.get("log_p_i", np.nan),
                "cross_elasticity": fit.params.get("log_p_j", np.nan),
                "cross_elasticity_ci_low": ci.loc["log_p_j", 0] if "log_p_j" in ci.index else np.nan,
                "cross_elasticity_ci_high": ci.loc["log_p_j", 1] if "log_p_j" in ci.index else np.nan,
                "cross_elasticity_p_value": fit.pvalues.get("log_p_j", np.nan),
                "mae_val": mae_val,
                "rmse_val": rmse_val,
                "r2_val": r2_val,
            }

        except Exception as exc:
            # Return the error message
            return {
                "store_code": store_code,
                "pair_id": pair_id,
                "upc_i": upc_i,
                "upc_j": upc_j,
                "status": "error",
                "error_message": str(exc),
            }

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Columns needed for the model
        needed_cols = ["log_v_i", "log_p_i", "log_p_j"] + self.config.control_cols

        # Group the data by the group keys
        train_groups = {
            k: g[needed_cols].dropna()
            for k, g in train_df.groupby(self.GROUP_KEYS)
        }
        # Group the validation data by the group keys
        val_groups = {
            k: g[needed_cols].dropna()
            for k, g in val_df.groupby(self.GROUP_KEYS)
        }

        # Get the common keys between the train and validation groups
        common_keys = sorted(set(train_groups) & set(val_groups))
        results = []

        # Fit the model for each common key
        for key in common_keys:
            g_train = train_groups[key]
            g_val = val_groups[key]
            # Check if the group is valid
            if not self._is_valid(g_train, g_val):
                continue

            # Fit the model for the group
            row = self._fit_single(key, g_train, g_val)
            if row is not None:
                results.append(row)

        return pd.DataFrame(results)


