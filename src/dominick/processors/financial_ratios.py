import numpy as np
import pandas as pd

class FinancialRatiosCalculator:
    """
    Computes per-UPC financial metrics from raw Dominick's transaction data.
    Derived columns added to the DataFrame:
      - unit_price        : effective price per UPC within the deal (total_price / units_per_deal)
      - sales_dolar       : corrected revenue (total_price x units_sold / units_per_deal)
      - gross_margin_rate : gross margin as a fraction (gross_margin_pct / 100)
      - gross_margin_dolar: absolute gross margin in dollars
      - cost_dolar        : cost of goods sold (sales_dolar x (1 - gross_margin_rate))
    Required input columns: total_price, units_sold, units_per_deal, gross_margin_pct.
    Public API:
        run(df) -> pd.DataFrame
    """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public API. Adds financial columns.
        Requires: total_price, units_sold, units_per_deal, gross_margin_pct.
        """
        out = df.copy()
        denom = out["units_per_deal"].replace({0: np.nan})

        # Effective price per UPC within the deal
        out["unit_price"] = (out["total_price"] / denom).round(2)

        # Correct sales: Sales = Price * Move / Qty
        out["sales_dolar"] = (out["total_price"] * out["units_sold"] / denom).round(2)

        out["gross_margin_rate"] = out["gross_margin_pct"] / 100.0
        out["gross_margin_dolar"] = (out["sales_dolar"] * out["gross_margin_rate"]).round(2)
        out["cost_dolar"] = (out["sales_dolar"] * (1 - out["gross_margin_rate"])).round(2)
        return out