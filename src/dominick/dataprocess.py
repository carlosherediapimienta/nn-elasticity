import pandas as pd
from .processors import (
    FinancialRatiosCalculator,
    UnitConverter,
    LiterMetricsCalculator,
    ElasticityFeatureGenerator,
)


class DominickDataProcessor:
    """
    Orquestador de procesamiento de datos Dominick's.
    Coordina los procesadores usando su API pública (run).
    API pública: join(), rename(), add_ratios(), add_elasticity_features().
    """

    def __init__(
        self,
        financial_calculator: FinancialRatiosCalculator = None,
        unit_converter: UnitConverter = None,
        liter_calculator: LiterMetricsCalculator = None,
        elasticity_generator: ElasticityFeatureGenerator = None,
    ):
        self.financial_calculator = financial_calculator or FinancialRatiosCalculator()
        self.unit_converter = unit_converter or UnitConverter()
        self.liter_calculator = liter_calculator or LiterMetricsCalculator(self.unit_converter)
        self.elasticity_generator = elasticity_generator or ElasticityFeatureGenerator(self.liter_calculator)

    def join(self, left: pd.DataFrame, right: pd.DataFrame, on: str | list[str], how: str = "inner") -> pd.DataFrame:
        """API pública. Wrapper de pd.merge."""
        return pd.merge(left, right, on=on, how=how)

    def rename(self, df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
        """API pública. Wrapper de df.rename."""
        return df.rename(columns=columns)

    def add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        API pública. Añade ratios financieros.
        Delega en FinancialRatiosCalculator.run().
        """
        return self.financial_calculator.run(df)

    def add_elasticity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        API pública. Añade features de elasticidad.
        Delega en ElasticityFeatureGenerator.run().
        """
        return self.elasticity_generator.run(df)