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

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        formula = self.formula_builder.build_formula()
        results = []

        for keys, group_df in df.groupby([self.config.store_col, self.config.upc_col]):
            store_id, upc = keys

            model_df = self.dataset_builder.build(group_df)

            if len(model_df) < self.config.min_obs:
                continue

            model = LogLogBenchmarkFairModel(
                price_col=self.config.price_col,
                robust_cov_type=self.config.robust_cov_type,
            )

            try:
                model.fit(formula=formula, model_df=model_df)
                ci = model.confidence_interval_for_elasticity()
                results.append({
                    "store_code":          store_id,
                    "upc_code":            upc,
                    "elasticity":          model.elasticity(),
                    "elasticity_se":       model.result.bse[self.config.price_col],
                    "elasticity_ci_low":   ci["lower_95"],
                    "elasticity_ci_high":  ci["upper_95"],
                    "elasticity_p_value":  model.p_value_for_elasticity(),
                    **model.metrics(),
                })
            except Exception as e:
                results.append({
                    "store_code": store_id,
                    "upc_code":   upc,
                    "error":      str(e),
                })

        return pd.DataFrame(results)