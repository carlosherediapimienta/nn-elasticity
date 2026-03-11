from typing import List
import pandas as pd
from .config import ElasticityConfig


class BenchmarkFairFormulaBuilder:
    """Responsabilidad única: construir la fórmula statsmodels."""

    def __init__(self, config: ElasticityConfig) -> None:
        self.config = config

    def build_formula(self) -> str:
        rhs_terms: List[str] = [self.config.price_col]
        rhs_terms.extend(self.config.numeric_control_cols)

        if self.config.include_store_fixed_effect:
            rhs_terms.append(f"C({self.config.store_col})")

        if self.config.include_upc_fixed_effect:
            rhs_terms.append(f"C({self.config.upc_col})")

        if self.config.include_category_fixed_effect:
            rhs_terms.append(f"C({self.config.category_col})")

        rhs = " + ".join(rhs_terms)
        return f"{self.config.target_col} ~ {rhs}"


class ModelingDatasetBuilder:
    """Responsabilidad única: preparar el dataframe final para el ajuste."""

    def __init__(self, config: ElasticityConfig) -> None:
        self.config = config

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [self.config.target_col, self.config.price_col]
        cols += self.config.numeric_control_cols

        if self.config.include_store_fixed_effect:
            cols.append(self.config.store_col)

        if self.config.include_upc_fixed_effect:
            cols.append(self.config.upc_col)

        if self.config.include_category_fixed_effect:
            cols.append(self.config.category_col)

        return df[cols].copy().dropna()