import numpy as np
import pandas as pd
from .unit_converter import UnitConverter

class LiterMetricsCalculator:
    """
    Calcula métricas relacionadas con litros: liters_per_upc, liters_sold, price_per_liter, etc.
    API pública: run().
    """

    def __init__(self, unit_converter: UnitConverter = None):
        self.unit_converter = unit_converter or UnitConverter()

    def run(
        self,
        df: pd.DataFrame,
        size_col: str = "pack_size_text",
        liters_col: str = "liters_per_upc",
    ) -> pd.DataFrame:
        """
        API pública. Añade métricas de litros.
        Requiere: total_price, units_sold, units_per_deal, pack_size_text.
        """
        out = df.copy()
        denom = out["units_per_deal"].replace({0: np.nan})

        if liters_col not in out.columns:
            out[liters_col] = out[size_col].apply(self.unit_converter.run)

        out["price_per_upc"] = out["total_price"] / denom
        out["liters_sold"] = out["units_sold"] * out[liters_col]
        out["price_per_liter"] = out["price_per_upc"] / out[liters_col]

        if "sales_dolar" not in out.columns:
            out["sales_dolar"] = out["total_price"] * out["units_sold"] / denom

        out["avg_price_per_liter"] = out["sales_dolar"] / out["liters_sold"]

        # Redondeos
        out["price_per_upc"] = out["price_per_upc"].round(4)
        out["price_per_liter"] = out["price_per_liter"].round(4)
        out["avg_price_per_liter"] = out["avg_price_per_liter"].round(4)
        out["liters_sold"] = out["liters_sold"].round(6)
        out[liters_col] = out[liters_col].round(6)
        return out