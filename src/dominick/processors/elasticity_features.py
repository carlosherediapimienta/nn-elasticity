import numpy as np
import pandas as pd
from .liter_metrics import LiterMetricsCalculator

class ElasticityFeatureGenerator:
    """
    Genera features para estimación de elasticidad: logs y flags de promoción.
    API pública: run().
    """

    def __init__(self, liter_calculator: LiterMetricsCalculator = None):
        self.liter_calculator = liter_calculator or LiterMetricsCalculator()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        API pública. Añade features de elasticidad.
        Requiere: total_price, units_sold, units_per_deal, pack_size_text, promo_flag (opcional).
        """
        out = df.copy()

        # Ensure deal-corrected prices + liter metrics
        out = self.liter_calculator.run(out)

        # Promo helpers (SALE inconsistent; this is still useful)
        pf = out.get("promo_flag", pd.Series(index=out.index, dtype="object"))
        pf = pf.fillna("").astype(str).str.strip().str.upper()
        out["on_promo"] = pf.ne("")  # if set => promo; if empty, promo might still exist

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

        out["log_liters_sold"] = safe_log(out["liters_sold"])
        out["log_price_per_liter"] = safe_log(out["price_per_liter"])

        return out