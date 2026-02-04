import re
import numpy as np
import pandas as pd


class DominickDataProcessor:
    def __init__(self):
        self.OZ_TO_L = 0.0295735296      # US fluid oz -> liters
        self.GAL_TO_L = 3.785411784      # US gallon -> liters

    def join(self, left: pd.DataFrame, right: pd.DataFrame, on: str | list[str], how: str = "inner") -> pd.DataFrame:
        return pd.merge(left, right, on=on, how=how)

    def rename(self, df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=columns)

    def add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds: sales_dolar, unit_price, gross_margin_rate, gross_margin_dolar, cost_dolar.
        Requires: total_price, units_sold, units_per_deal, gross_margin_pct.
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

    # ---------- LITROS ----------
    def _normalize_pack_size(self, s: str) -> str:
        s = s.strip().upper()
        s = s.replace(" ", "").replace("OZ.", "OZ")
        if s.endswith("."):
            s = s[:-1]

        # "16.9O" -> "16.9OZ"
        s = re.sub(r'(?<=\d)O$', 'OZ', s)
        # "12/12O" -> "12/12OZ"
        s = re.sub(r'/(\d+(?:\.\d+)?)O$', r'/\1OZ', s)
        # "5.16GA" -> "5.16GAL"
        s = s.replace("GA", "GAL")
        return s

    def _to_liters_single(self, pack_size_text: str) -> float:
        """Return liters per UPC (liters_per_upc)."""
        if pack_size_text is None or (isinstance(pack_size_text, float) and np.isnan(pack_size_text)):
            return np.nan

        s = self._normalize_pack_size(str(pack_size_text))
        if not s:
            return np.nan

        # multipack: N/SIZEUNIT (e.g., 6/12OZ)
        m = re.match(r'^(?P<n>\d+)\/(?P<size>\d+(?:\.\d+)?)(?P<unit>[A-Z]+)$', s)
        if m:
            n = int(m.group("n"))
            size = float(m.group("size"))
            unit = m.group("unit")
            if unit == "ML":
                return n * (size / 1000.0)
            if unit == "OZ":
                return n * (size * self.OZ_TO_L)
            if unit in ("GAL", "GALLON", "GALLONS"):
                return n * (size * self.GAL_TO_L)
            return np.nan

        # single: SIZEUNIT (e.g., 750ML, 32OZ)
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
            return np.nan

        return np.nan

    def _add_liter_metrics(
        self,
        df: pd.DataFrame,
        size_col: str = "pack_size_text",
        liters_col: str = "liters_per_upc",
    ) -> pd.DataFrame:
        """
        Adds:
          - liters_per_upc
          - price_per_upc (effective, deal-corrected)
          - liters_sold
          - price_per_liter
          - avg_price_per_liter (realized: sales_dolar / liters_sold)
        Requires: total_price, units_sold, units_per_deal, pack_size_text
        """
        out = df.copy()
        denom = out["units_per_deal"].replace({0: np.nan})

        if liters_col not in out.columns:
            out[liters_col] = out[size_col].apply(self._to_liters_single)

        out["price_per_upc"] = out["total_price"] / denom
        out["liters_sold"] = out["units_sold"] * out[liters_col]
        out["price_per_liter"] = out["price_per_upc"] / out[liters_col]

        if "sales_dolar" not in out.columns:
            out["sales_dolar"] = out["total_price"] * out["units_sold"] / denom

        out["avg_price_per_liter"] = out["sales_dolar"] / out["liters_sold"]

        # Redondeos (opcionales)
        out["price_per_upc"] = out["price_per_upc"].round(4)
        out["price_per_liter"] = out["price_per_liter"].round(4)
        out["avg_price_per_liter"] = out["avg_price_per_liter"].round(4)
        out["liters_sold"] = out["liters_sold"].round(6)
        out[liters_col] = out[liters_col].round(6)
        return out

    # ---------- ELASTICITY (QUANTITY AND LITERS) ----------
    def add_elasticity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds minimal features for elasticity estimation (log-log), both by:
        - UPC quantity: units_sold vs price_per_upc
        - Liters quantity: liters_sold vs price_per_liter

        Adds:
          Quantity/Price:
            - price_per_upc
            - liters_per_upc
            - liters_sold
            - price_per_liter

          Logs (NaN if non-positive):
            - log_units_sold
            - log_price_per_upc
            - log_liters_sold
            - log_price_per_liter

          Promo helpers:
            - on_promo (best-effort; SALE can be inconsistent)
            - promo_B, promo_C, promo_S

        Notes:
          - For true log-log elasticity you typically drop rows with Q<=0 or P<=0.
          - Exclude flagged rows using exclude_flag outside (or drop here if you want).
        """
        out = df.copy()

        # Ensure deal-corrected prices + liter metrics
        out = self._add_liter_metrics(out)

        # Promo helpers (SALE inconsistent; this is still useful)
        pf = out.get("promo_flag", pd.Series(index=out.index, dtype="object"))
        pf = pf.fillna("").astype(str).str.strip().str.upper()
        out["on_promo"] = pf.ne("")  # if set => promo; if empty, promo might still exist (manual warning)

        out["promo_B"] = (pf == "B").astype(int)
        out["promo_C"] = (pf == "C").astype(int)
        out["promo_S"] = (pf == "S").astype(int)

        # Logs (guard against 0/negative)
        def safe_log(x: pd.Series) -> pd.Series:
            x = pd.to_numeric(x, errors="coerce").astype(float)
            # inf/-inf -> NaN
            x = x.replace([np.inf, -np.inf], np.nan)
            # <=0 -> NaN (avoids log(0))
            x = x.mask(x <= 0, np.nan)
            return np.log(x)

        out["log_units_sold"] = safe_log(out["units_sold"])
        out["log_price_per_upc"] = safe_log(out["price_per_upc"])

        out["log_liters_sold"] = safe_log(out["liters_sold"])
        out["log_price_per_liter"] = safe_log(out["price_per_liter"])

        return out