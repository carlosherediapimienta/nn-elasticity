import pandas as pd
import statsmodels.api as sm
from .config import ElasticityConfig

class DesignMatrixBuilder:
    """Se encarga solo de construir X e y."""

    def __init__(self, config: ElasticityConfig) -> None:
        self.config = config

    def build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        cols = [self.config.price_col] + self.config.control_cols
        clean_df = df[[self.config.target_col] + cols].dropna().copy()

        X = clean_df[cols]
        X = sm.add_constant(X, has_constant="add")
        y = clean_df[self.config.target_col]
        return X, y