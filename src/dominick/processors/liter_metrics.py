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
        denom = df["units_per_deal"].replace({0: np.nan})

        if liters_col not in df.columns:
            # Convert pack_size_text to liters per UPC.
            df[liters_col] = df[size_col].apply(self.unit_converter.run)

        # Effective price per UPC.
        df["price_per_upc"] = df["total_price"] / denom
        # Total liters sold.
        df["liters_sold"] = df["units_sold"] * df[liters_col]
        df["price_per_liter"] = df["price_per_upc"] / df[liters_col]

        if "sales_dolar" not in df.columns:
            df["sales_dolar"] = df["total_price"] * df["units_sold"] / denom

        # Rounding.
        df["price_per_upc"] = df["price_per_upc"].round(4)
        df["price_per_liter"] = df["price_per_liter"].round(4)
        df["liters_sold"] = df["liters_sold"].round(6)
        df[liters_col] = df[liters_col].round(6)
        return df