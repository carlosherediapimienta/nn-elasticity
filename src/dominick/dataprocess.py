import pandas as pd


class DominickDataProcessor:
    def __init__(self):
        pass

    def join(self, left: pd.DataFrame, right: pd.DataFrame, on: str | list[str], how: str = "inner") -> pd.DataFrame:
        return pd.merge(left, right, on=on, how=how)

    def rename(self, df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=columns)

    def add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds: sales_dolar, unit_price, gross_margin_rate, gross_margin_dolar, cost_dolar.
        Requiere columnas: total_price, units_sold, units_per_deal, gross_margin_pct.
        """
        out = df.copy()
        out["unit_price"] = out["total_price"] / out["units_per_deal"]
        out["sales_dolar"] = out["total_price"] * out["units_sold"] / out["units_per_deal"]
        out["gross_margin_rate"] = out["gross_margin_pct"] / 100
        out["gross_margin_dolar"] = out["sales_dolar"] * out["gross_margin_rate"]
        out["cost_dolar"] = out["sales_dolar"] * (1 - out["gross_margin_rate"])
        return out