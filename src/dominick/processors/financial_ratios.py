import numpy as np
import pandas as pd

class FinancialRatiosCalculator:
    """
    Calcula ratios financieros: sales_dolar, unit_price, gross_margin, cost_dolar.
    API pública: run().
    """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        API pública. Añade columnas financieras.
        Requiere: total_price, units_sold, units_per_deal, gross_margin_pct.
        """
        out = df.copy()
        denom = out["units_per_deal"].replace({0: np.nan})

        # Precio efectivo por UPC dentro del deal
        out["unit_price"] = (out["total_price"] / denom).round(2)

        # Ventas $ correctas: Sales = Price * Move / Qty
        out["sales_dolar"] = (out["total_price"] * out["units_sold"] / denom).round(2)

        out["gross_margin_rate"] = out["gross_margin_pct"] / 100.0
        out["gross_margin_dolar"] = (out["sales_dolar"] * out["gross_margin_rate"]).round(2)
        out["cost_dolar"] = (out["sales_dolar"] * (1 - out["gross_margin_rate"])).round(2)
        return out