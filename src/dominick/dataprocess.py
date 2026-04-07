import pandas as pd
from .processors import (
    FinancialRatiosCalculator,
    UnitConverter,
    LiterMetricsCalculator,
    ElasticityFeatureGenerator,
)

class DominickDataProcessor:
    """
    Orchestrator for Dominick's raw data processing pipeline.

    Coordinates four specialized processors via dependency injection,
    each exposing a single .run(df) method:
      - FinancialRatiosCalculator   : margin and price ratios
      - UnitConverter               : unit standardization
      - LiterMetricsCalculator      : volume metrics in liters
      - ElasticityFeatureGenerator  : log-price and log-demand features

    Public API:
        join(left, right, on, how)   : pd.DataFrame
        rename(df, columns)          : pd.DataFrame
        add_ratios(df)               : pd.DataFrame
        add_elasticity_features(df)  : pd.DataFrame
    """

    def __init__(self):
        self.financial_calculator = FinancialRatiosCalculator()
        self.unit_converter = UnitConverter()
        self.liter_calculator = LiterMetricsCalculator(unit_converter=self.unit_converter)
        self.elasticity_generator = ElasticityFeatureGenerator(self.liter_calculator)

    def join(self, left: pd.DataFrame, right: pd.DataFrame, on: str | list[str], how: str = "inner") -> pd.DataFrame:
        """Public API. Wrapper of pd.merge."""
        return pd.merge(left, right, on=on, how=how)

    def rename(self, df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
        """Public API. Wrapper of df.rename."""
        return df.rename(columns=columns)

    def add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public API. Adds financial ratios.
        Delegates to FinancialRatiosCalculator.run().
        """
        return self.financial_calculator.run(df)

    def add_elasticity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public API. Adds elasticity features.
        Delegates to ElasticityFeatureGenerator.run().
        """
        return self.elasticity_generator.run(df)