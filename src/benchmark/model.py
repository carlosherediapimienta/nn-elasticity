from typing import Dict
import pandas as pd
import statsmodels.formula.api as smf

class LogLogBenchmarkModel:
    """Responsabilidad única: ajustar y exponer resultados del modelo OLS."""

    def __init__(self, formula: str) -> None:
        self.formula = formula
        self.result = None

    def fit(self, train_df: pd.DataFrame, cov_type: str = "HC1") -> None:
        self.result = smf.ols(formula=self.formula, data=train_df).fit(cov_type=cov_type)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        return self.result.predict(df)

    def elasticity(self, price_col: str) -> float:
        self._check_fitted()
        return float(self.result.params[price_col])

    def confidence_interval_for_elasticity(self, price_col: str) -> Dict[str, float]:
        self._check_fitted()
        ci = self.result.conf_int().loc[price_col]
        return {
            "lower_95": float(ci.iloc[0]),
            "upper_95": float(ci.iloc[1]),
        }

    def p_value_for_elasticity(self, price_col: str) -> float:
        self._check_fitted()
        return float(self.result.pvalues[price_col])

    def coefficient_for(self, col: str) -> float:
        self._check_fitted()
        coef = self.result.params.get(col)
        if coef is None or pd.isna(coef):
            raise ValueError(f"Coefficient '{col}' not available.")
        return float(coef)

    def confidence_interval_for(self, col: str) -> Dict[str, float]:
        self._check_fitted()
        ci = self.result.conf_int().loc[col]
        return {
            "lower_95": float(ci.iloc[0]),
            "upper_95": float(ci.iloc[1]),
        }

    def p_value_for(self, col: str) -> float:
        self._check_fitted()
        return float(self.result.pvalues[col])

    def _check_fitted(self) -> None:
        if self.result is None:
            raise RuntimeError("Model has not been fitted yet.")