import numpy as np
import pandas as pd

class LiterMetricsCalculator:
    """
    Computes volume-based metrics from Dominick's transaction data.
    Delegates pack_size_text to liters conversion to UnitConverter, then
    derives the following columns:
      - liters_per_upc  : liters per individual UPC (via UnitConverter)
      - price_per_upc   : effective price per UPC (total_price / units_per_deal)
      - liters_sold     : total liters sold (units_sold x liters_per_upc)
      - price_per_liter : price per liter (price_per_upc / liters_per_upc)
      - sales_dolar     : corrected revenue, added only if not already present
    Required input columns: total_price, units_sold, units_per_deal, pack_size_text.
    Public API:
        run(df, size_col, liters_col) -> pd.DataFrame
    """
    def __init__(self, unit_converter):
        self.unit_converter = unit_converter

    def run(
        self,
        df: pd.DataFrame,
        size_col: str = "pack_size_text",
        liters_col: str = "liters_per_upc",
    ) -> pd.DataFrame:
        """
        Public API. Adds volume metrics.
        Requires: total_price, units_sold, units_per_deal, pack_size_text.
        """
        out = df.copy()
        denom = out["units_per_deal"].replace({0: np.nan})

        if liters_col not in out.columns:
            # Convert pack_size_text to liters per UPC.
            out[liters_col] = out[size_col].apply(self.unit_converter.run)

        # Effective price per UPC.
        out["price_per_upc"] = out["total_price"] / denom
        # Total liters sold.
        out["liters_sold"] = out["units_sold"] * out[liters_col]
        out["price_per_liter"] = out["price_per_upc"] / out[liters_col]

        if "sales_dolar" not in out.columns:
            out["sales_dolar"] = out["total_price"] * out["units_sold"] / denom

        # Rounding.
        out["price_per_upc"] = out["price_per_upc"].round(4)
        out["price_per_liter"] = out["price_per_liter"].round(4)
        out["liters_sold"] = out["liters_sold"].round(6)
        out[liters_col] = out[liters_col].round(6)
        return out