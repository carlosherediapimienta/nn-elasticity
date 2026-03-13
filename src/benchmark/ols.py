import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .builder import BenchmarkFormulaBuilder, ModelingDatasetBuilder
from .config import ElasticityConfig
from .model import LogLogBenchmarkModel


class ElasticityBenchmarkPipeline:
    """Pipeline benchmark OLS log-log por store x UPC."""

    def __init__(self, config: ElasticityConfig) -> None:
        self.config = config
        self.formula_builder = BenchmarkFormulaBuilder(config)
        self.dataset_builder = ModelingDatasetBuilder(config)

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> pd.DataFrame:
        results = []

        train_groups = {
            key: group.copy()
            for key, group in train_df.groupby([self.config.store_col, self.config.upc_col])
        }
        val_groups = {
            key: group.copy()
            for key, group in val_df.groupby([self.config.store_col, self.config.upc_col])
        }

        common_keys = sorted(set(train_groups.keys()) & set(val_groups.keys()))
        formula = self.formula_builder.build_formula()

        for (store_id, upc) in common_keys:
            train_group = train_groups[(store_id, upc)].copy()
            val_group = val_groups[(store_id, upc)].copy()

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
                    "status": "empty_val",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })
                continue

            if train_model_df[self.config.price_col].nunique() < 2:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "no_price_variation",
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })
                continue

            invalid_cross = False
            for cross_col in self.config.cross_price_cols:
                if cross_col not in train_model_df.columns:
                    results.append({
                        self.config.store_col: store_id,
                        self.config.upc_col: upc,
                        "status": "missing_cross_price_col",
                        "missing_cross_col": cross_col,
                        "n_train": int(len(train_model_df)),
                        "n_val": int(len(val_model_df)),
                    })
                    invalid_cross = True
                    break

                if train_model_df[cross_col].nunique() < 2:
                    results.append({
                        self.config.store_col: store_id,
                        self.config.upc_col: upc,
                        "status": "no_cross_price_variation",
                        "cross_price_col": cross_col,
                        "n_train": int(len(train_model_df)),
                        "n_val": int(len(val_model_df)),
                    })
                    invalid_cross = True
                    break

            if invalid_cross:
                continue

            model = LogLogBenchmarkModel(formula=formula)

            try:
                model.fit(train_model_df, cov_type=self.config.robust_cov_type)

                val_pred = model.predict(val_model_df)

                mae_val = mean_absolute_error(
                    val_model_df[self.config.target_col],
                    val_pred,
                )
                rmse_val = mean_squared_error(
                    val_model_df[self.config.target_col],
                    val_pred,
                    squared=False,
                )
                r2_val = r2_score(
                    val_model_df[self.config.target_col],
                    val_pred,
                )

                coef = model.result.params.get(self.config.price_col)
                se = model.result.bse.get(self.config.price_col)
                p_value = model.result.pvalues.get(self.config.price_col)

                if pd.isna(coef) or pd.isna(se) or pd.isna(p_value):
                    results.append({
                        self.config.store_col: store_id,
                        self.config.upc_col: upc,
                        "status": "invalid_elasticity",
                        "n_train": int(len(train_model_df)),
                        "n_val": int(len(val_model_df)),
                    })
                    continue

                ci = model.confidence_interval_for_elasticity(self.config.price_col)

                cross_metrics = {}

                for cross_col in self.config.cross_price_cols:
                    try:
                        cross_coef = model.coefficient_for(cross_col)
                        cross_ci = model.confidence_interval_for(cross_col)
                        cross_p_value = model.p_value_for(cross_col)
                        cross_se = float(model.result.bse[cross_col])

                        cross_metrics[f"{cross_col}_elasticity"] = cross_coef
                        cross_metrics[f"{cross_col}_elasticity_se"] = cross_se
                        cross_metrics[f"{cross_col}_elasticity_ci_low"] = cross_ci["lower_95"]
                        cross_metrics[f"{cross_col}_elasticity_ci_high"] = cross_ci["upper_95"]
                        cross_metrics[f"{cross_col}_elasticity_p_value"] = cross_p_value

                    except Exception:
                        cross_metrics[f"{cross_col}_elasticity"] = np.nan
                        cross_metrics[f"{cross_col}_elasticity_se"] = np.nan
                        cross_metrics[f"{cross_col}_elasticity_ci_low"] = np.nan
                        cross_metrics[f"{cross_col}_elasticity_ci_high"] = np.nan
                        cross_metrics[f"{cross_col}_elasticity_p_value"] = np.nan

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
                    "mae_val": float(mae_val),
                    "rmse_val": float(rmse_val),
                    "r2_val": float(r2_val),
                    **cross_metrics,
                })

            except Exception as exc:
                results.append({
                    self.config.store_col: store_id,
                    self.config.upc_col: upc,
                    "status": "error",
                    "error_message": str(exc),
                    "n_train": int(len(train_model_df)),
                    "n_val": int(len(val_model_df)),
                })

        return pd.DataFrame(results)