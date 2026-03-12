import numpy as np
import pandas as pd
from typing import Dict, Any

from .builder import ModelingDatasetBuilder, BenchmarkFairFormulaBuilder
from .model import LogLogBenchmarkFairModel
from .config import ElasticityConfig


class ElasticityBenchmarkFairPipeline:
    """Responsabilidad única: orquestar el flujo completo."""

    def __init__(
        self,
        config: ElasticityConfig,
        dataset_builder: ModelingDatasetBuilder,
        formula_builder: BenchmarkFairFormulaBuilder,
    ) -> None:
        self.config = config
        self.dataset_builder = dataset_builder
        self.formula_builder = formula_builder

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        formula = self.formula_builder.build_formula()
        results = []

        train_groups = train_df.groupby([self.config.store_col, self.config.upc_col])
        val_groups = val_df.groupby([self.config.store_col, self.config.upc_col])

        train_group_dict = {k: g for k, g in train_groups}
        val_group_dict = {k: g for k, g in val_groups}

        common_keys = sorted(set(train_group_dict.keys()) & set(val_group_dict.keys()))

        for keys in common_keys:
            store_id, upc = keys

            train_group = train_group_dict[keys]
            val_group = val_group_dict[keys]

            train_model_df = self.dataset_builder.build(train_group)
            val_model_df = self.dataset_builder.build(val_group)

            if len(train_model_df) < self.config.min_obs:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "insufficient_train_obs",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })
                continue

            if len(val_model_df) == 0:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "empty_val_after_dropna",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })
                continue

            if train_model_df[self.config.price_col].nunique() < 2:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "no_price_variation_train",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })
                continue

            if train_model_df[self.config.price_col].var() == 0:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "zero_price_variance_train",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })
                continue

            model = LogLogBenchmarkFairModel(
                price_col=self.config.price_col,
                robust_cov_type=self.config.robust_cov_type,
            )

            try:
                model.fit(formula=formula, model_df=train_model_df)

                coef = model.result.params.get(self.config.price_col)
                se = model.result.bse.get(self.config.price_col)
                p_value = model.result.pvalues.get(self.config.price_col)

                if pd.isna(coef) or pd.isna(se) or pd.isna(p_value):
                    results.append({
                        self.config.store_col: store_id,
                        self.config.upc_col: upc,
                        "status": "invalid_fit_nan",
                        "n_train": int(len(train_model_df)),
                        "n_val": int(len(val_model_df)),
                    })
                    continue

                if not np.isfinite(coef) or not np.isfinite(se):
                    results.append({
                        self.config.store_col: store_id,
                        self.config.upc_col: upc,
                        "status": "invalid_fit_non_finite",
                        "n_train": int(len(train_model_df)),
                        "n_val": int(len(val_model_df)),
                    })
                    continue

                y_true = val_model_df[self.config.target_col]
                y_pred = model.predict(val_model_df)

                mae_val = float(np.mean(np.abs(y_true - y_pred)))
                rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
                r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

                ci = model.confidence_interval_for_elasticity()

                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "ok",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                    "elasticity": float(coef),
                    "elasticity_se": float(se),
                    "elasticity_ci_low": ci["lower_95"],
                    "elasticity_ci_high": ci["upper_95"],
                    "elasticity_p_value": float(p_value),
                    "mae_val": mae_val,
                    "rmse_val": rmse_val,
                    "r2_val": r2_val,
                })

            except Exception as e:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "fit_or_predict_error",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                    "error": str(e),
                })

        return pd.DataFrame(results)