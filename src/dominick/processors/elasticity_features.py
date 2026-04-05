import numpy as np
import pandas as pd

class ElasticityFeatureGenerator:
    """
    Generates features required for demand and elasticity estimation.
    Delegates volume metric computation to LiterMetricsCalculator, then adds:
      - on_promo        : bool flag, True if any promotion is active
      - promo_B/C/S     : binary indicators for each Dominick's promo type (B, C, S)
      - log_liters_sold : log of total liters sold (NaN-safe, guards against 0/negative)
      - log_price_per_liter: log of price per liter (NaN-safe)
    Note: promo_flag is optional; if absent, all promo columns default to False/0.
    The SALE flag in Dominick's data is inconsistent, so only B/C/S are encoded.
    Required input columns: total_price, units_sold, units_per_deal, pack_size_text.
    Optional input columns: promo_flag.
    Public API:
        run(df) -> pd.DataFrame
    """

    def __init__(self, liter_calculator):
        self.liter_calculator = liter_calculator

    def _safe_log(self, x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").astype(float)
        # inf/-inf -> NaN
        x = x.replace([np.inf, -np.inf], np.nan)
        # <=0 -> NaN (avoids log(0))
        x = x.mask(x <= 0, np.nan)
        return np.log(x)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public API. Adds elasticity features.
        Requires: total_price, units_sold, units_per_deal, pack_size_text, promo_flag (optional).
        """
        out = df.copy()

        # Ensure deal-corrected prices + liter metrics
        out = self.liter_calculator.run(out)

        # Promo helpers (SALE inconsistent; this is still useful)
        # We get the promo_flag column from the DataFrame.
        # and fill missing values with empty strings.
        pf = out.get("promo_flag", pd.Series(index=out.index, dtype="object"))
        pf = pf.fillna("").astype(str).str.strip().str.upper()
        # We create the on_promo column.
        out["on_promo"] = pf.ne("")

        # We create the promo_B, promo_C, promo_S columns.
        out["promo_B"] = (pf == "B").astype(int)
        out["promo_C"] = (pf == "C").astype(int)
        out["promo_S"] = (pf == "S").astype(int)

        # Logs (guard against 0/negative)
        out["log_liters_sold"] = self._safe_log(out["liters_sold"])
        out["log_price_per_liter"] = self._safe_log(out["price_per_liter"])

        return out