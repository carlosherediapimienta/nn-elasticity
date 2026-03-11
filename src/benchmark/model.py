import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any

class LogLogElasticityModel:
    """Se encarga solo de ajustar el modelo y devolver resultados."""

    def __init__(self, price_col: str) -> None:
        self.price_col = price_col
        self.result = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        model = sm.OLS(y, X)
        self.result = model.fit()

    def elasticity(self) -> float:
        if self.result is None:
            raise RuntimeError("Primero debes ajustar el modelo.")
        return float(self.result.params[self.price_col])

    def metrics(self) -> Dict[str, Any]:
        if self.result is None:
            raise RuntimeError("Primero debes ajustar el modelo.")
        return {
            "r_squared": float(self.result.rsquared),
            "adj_r_squared": float(self.result.rsquared_adj),
            "n_obs": int(self.result.nobs),
        }

    def summary_text(self) -> str:
        if self.result is None:
            raise RuntimeError("Primero debes ajustar el modelo.")
        return self.result.summary().as_text()