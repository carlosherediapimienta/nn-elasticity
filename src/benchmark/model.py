import statsmodels.formula.api as smf
import pandas as pd
from typing import Dict, Any


class LogLogBenchmarkFairModel:
    """
    Responsabilidad única: ajustar el modelo y exponer resultados.

    En un modelo log-log:
    - target = log(cantidad)
    - price  = log(precio)

    El coeficiente de price se interpreta como elasticidad precio condicional.
    """

    def __init__(self, price_col: str, robust_cov_type: str = "HC1") -> None:
        self.price_col = price_col
        self.robust_cov_type = robust_cov_type
        self.result = None

    def fit(self, formula: str, model_df: pd.DataFrame) -> None:
        model = smf.ols(formula=formula, data=model_df)
        self.result = model.fit(cov_type=self.robust_cov_type)

    def predict(self, model_df: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        return self.result.predict(model_df)

    def elasticity(self) -> float:
        self._check_fitted()
        coef = self.result.params.get(self.price_col)
        if coef is None or pd.isna(coef):
            raise ValueError(f"Elasticity coefficient '{self.price_col}' not available.")
        return float(coef)

    def confidence_interval_for_elasticity(self) -> Dict[str, float]:
        self._check_fitted()
        ci = self.result.conf_int().loc[self.price_col]
        return {
            "lower_95": float(ci.iloc[0]),
            "upper_95": float(ci.iloc[1]),
        }

    def p_value_for_elasticity(self) -> float:
        self._check_fitted()
        return float(self.result.pvalues[self.price_col])

    def metrics(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "n_obs":         int(self.result.nobs),
            "r_squared":     float(self.result.rsquared),
            "adj_r_squared": float(self.result.rsquared_adj),
            "aic":           float(self.result.aic),
            "bic":           float(self.result.bic),
        }

    def summary_text(self) -> str:
        self._check_fitted()
        return self.result.summary().as_text()

    def _check_fitted(self) -> None:
        if self.result is None:
            raise RuntimeError("El modelo todavía no ha sido ajustado.")