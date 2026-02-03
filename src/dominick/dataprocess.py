import pandas as pd
import re


class DominickDataProcessor:
    def __init__(self):
        self.OZ_TO_L = 0.0295735296
        self.GAL_TO_L = 2.785411784

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

    def _normalize_pack_size(self, s: str) -> str:
        s = s.strip().upper()
        s = s.replace(" ", "")
        s = s.replace("OZ.", "OZ")
        s = re.sub(r'(\d)(O)$', r'\1OZ', s)
        s = re.sub(r'(\d)O$', r'\1OZ', s)
        s = re.sub(r'/(\d+(?:\.\d+)?)O$', r'/\1OZ', s)
        s = s.replace("GA", "GAL")
        return s

    def _to_liters_single(self, pack_size_text: str) -> float | None:
        if pd.isna(pack_size_text) or not isinstance(pack_size_text, str):
            return None
        s = self._normalize_pack_size(str(pack_size_text))
        # multipack: N/SIZEUNIT
        m = re.match(r'^(?P<n>\d+)\/(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)?$', s)
        if m:
            n = int(m.group("n"))
            size = float(m.group("size"))
            unit = (m.group("unit") or "")
            if unit == "ML":
                return n * (size / 1000.0)
            if unit == "OZ":
                return n * (size * self.OZ_TO_L)
            if unit in ("GAL", "GALLON", "GALLONS"):
                return n * (size * self.GAL_TO_L)
            return None
        # single: SIZEUNIT
        m = re.match(r'^(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)$', s)
        if m:
            size = float(m.group("size"))
            unit = m.group("unit")
            if unit == "ML":
                return size / 1000.0
            if unit == "OZ":
                return size * self.OZ_TO_L
            if unit in ("GAL", "GALLON", "GALLONS"):
                return size * self.GAL_TO_L
            return None
        return None

    def to_liters(self, df: pd.DataFrame, text_column: str, output_column: str = "volume_liters") -> pd.DataFrame:
        out = df.copy()
        out[output_column] = out[text_column].map(self._to_liters_single)
        return out

