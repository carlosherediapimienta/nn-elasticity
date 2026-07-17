import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .config import RidgeConfig


class RegularizedElasticityPipeline:
    """
    Regularized (Ridge) log-log benchmark for own- and cross-price
    elasticities. Mirrors PairwiseElasticityPipeline EXACTLY: same dataset,
    same grouping by (store_code, pair_id, upc_i, upc_j), same formula
    log_v_i ~ log_p_i + log_p_j + controls -- so it plugs directly into the
    same downstream analysis (analysis-results.ipynb) as the OLS benchmark.
    The ONLY difference is the estimator: RidgeCV (alpha chosen by internal
    CV) instead of unpenalized OLS. This isolates the effect of coefficient
    shrinkage alone (pooling is held fixed at "none", same as the OLS).
    Groups with insufficient price variation or too few observations are
    silently skipped, mirroring PairwiseElasticityPipeline.
    """
    GROUP_KEYS = ["store_code", "pair_id", "upc_i", "upc_j"]

    def __init__(self, config: RidgeConfig) -> None:
        self.config = config

    def _is_valid(self, g_train: pd.DataFrame, g_val: pd.DataFrame) -> bool:
        if len(g_train) < self.config.min_obs:
            return False
        if len(g_val) == 0:
            return False
        if g_train["log_p_i"].nunique() < 2:
            return False
        if g_train["log_p_j"].nunique() < 2:
            return False
        return True

    def _fit_single(
        self, key: tuple, g_train: pd.DataFrame, g_val: pd.DataFrame
    ) -> tuple[dict | None, pd.DataFrame | None]:
        store_code, pair_id, upc_i, upc_j = key
        feature_cols = ["log_p_i", "log_p_j"] + self.config.control_cols

        try:
            scaler = StandardScaler().fit(g_train[feature_cols]) if self.config.standardize else None
            Xtr = scaler.transform(g_train[feature_cols]) if scaler else g_train[feature_cols].values
            Xval = scaler.transform(g_val[feature_cols]) if scaler else g_val[feature_cols].values
            ytr, yval = g_train["log_v_i"], g_val["log_v_i"]

            model = RidgeCV(alphas=self.config.alphas, cv=self.config.cv_folds).fit(Xtr, ytr)
            pred_val = model.predict(Xval)

            residuals = yval.values - pred_val
            mae_val = np.mean(np.abs(residuals))
            rmse_val = np.sqrt(np.mean(residuals ** 2))
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((yval.values - yval.mean()) ** 2)
            r2_val = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

            scale = scaler.scale_ if scaler else np.ones(len(feature_cols))
            coefs = dict(zip(feature_cols, model.coef_ / scale))

            summary = {
                "store_code": store_code,
                "pair_id": pair_id,
                "upc_i": upc_i,
                "upc_j": upc_j,
                "status": "ok",
                "n_train": len(g_train),
                "n_val": len(g_val),
                "alpha_selected": model.alpha_,
                "own_elasticity": coefs["log_p_i"],
                "cross_elasticity": coefs["log_p_j"],
                "mae_val": mae_val,
                "rmse_val": rmse_val,
                "r2_val": r2_val,
            }

            # Row-level predictions, keyed by week. Needed to later collapse
            # across partner UPCs j and pool residuals on a common
            # (store, upc_i, week) unit, comparable to ICDN's global metric.
            predictions = pd.DataFrame({
                "store_code": store_code,
                "pair_id": pair_id,
                "upc_i": upc_i,
                "upc_j": upc_j,
                "week_id": g_val["week_id"].values,
                "y_true_i": yval.values,
                "y_hat_i": pred_val,
            })

            return summary, predictions
        except Exception as exc:
            return {
                "store_code": store_code,
                "pair_id": pair_id,
                "upc_i": upc_i,
                "upc_j": upc_j,
                "status": "error",
                "error_message": str(exc),
            }, None

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # +week_id, kept only as a row key, never used as a regressor.
        needed_cols = ["log_v_i", "log_p_i", "log_p_j", "week_id"] + self.config.control_cols

        train_groups = {
            k: g[needed_cols].dropna() for k, g in train_df.groupby(self.GROUP_KEYS)
        }
        val_groups = {
            k: g[needed_cols].dropna() for k, g in val_df.groupby(self.GROUP_KEYS)
        }
        common_keys = sorted(set(train_groups) & set(val_groups))

        results = []
        pred_frames = []
        for key in common_keys:
            g_train = train_groups[key]
            g_val = val_groups[key]
            if not self._is_valid(g_train, g_val):
                continue
            row, preds = self._fit_single(key, g_train, g_val)
            if row is not None:
                results.append(row)
            if preds is not None:
                pred_frames.append(preds)

        summary_df = pd.DataFrame(results)
        predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        return summary_df, predictions_df