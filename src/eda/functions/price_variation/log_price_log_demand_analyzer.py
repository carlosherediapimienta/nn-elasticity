import pandas as pd
import numpy as np
from scipy import stats


class LogPriceLogDemandAnalyzer:
    """
    Sanity check economic: correlation of log(price)–log(demand)
    in three scenarios (global, within store×UPC, within non‑promo).
    NO causal, but useful to check for negative signal and the role of promo.
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        price_col: str = "log_price_per_liter",
        demand_col: str = "log_liters_sold",
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        promo_col: str = "on_promo",
    ) -> dict:
        """
        Args:
            df: DataFrame with grain (store, upc, week) and price, demand, promo columns.
            price_col: price column in log.
            demand_col: demand column in log.
            store_col, upc_col: grouping columns for “within”.
            promo_col: promo column (0/1) for filtering non‑promo.

        Returns:
            dict with:
                - corr_global: Spearman global correlation (all rows).
                - corr_within_store_upc: correlation on series demeaned by (store, upc).
                - corr_global_non_promo: global correlation restricted to rows with promo=0.
                - n_obs, n_obs_non_promo (counts).
                - pvalue for each correlation (optional).
        """
        for col in [price_col, demand_col, store_col, upc_col, promo_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        df_work = df[[store_col, upc_col, price_col, demand_col, promo_col]].dropna(
            subset=[price_col, demand_col]
        )
        n_obs = len(df_work)

        # --- Global ---
        r_global, p_global = stats.spearmanr(df_work[price_col], df_work[demand_col])

        # --- Within (demean by store×UPC) ---
        df_work = df_work.copy()
        df_work["_p_dm"] = df_work[price_col] - df_work.groupby([store_col, upc_col])[price_col].transform("mean")
        df_work["_d_dm"] = df_work[demand_col] - df_work.groupby([store_col, upc_col])[demand_col].transform("mean")
        r_within, p_within = stats.spearmanr(df_work["_p_dm"], df_work["_d_dm"])

        # --- Within non‑promo (global on rows with promo=0) ---
        non_promo = df_work[df_work[promo_col].eq(0)]
        n_obs_non_promo = len(non_promo)
        if n_obs_non_promo < 3:
            r_non_promo, p_non_promo = np.nan, np.nan
        else:
            r_non_promo, p_non_promo = stats.spearmanr(non_promo[price_col], non_promo[demand_col])

        return {
            "corr_global": float(r_global),
            "pvalue_global": float(p_global),
            "corr_within_store_upc": float(r_within),
            "pvalue_within_store_upc": float(p_within),
            "corr_global_non_promo": float(r_non_promo),
            "pvalue_global_non_promo": float(p_non_promo) if n_obs_non_promo >= 3 else None,
            "n_obs": n_obs,
            "n_obs_non_promo": n_obs_non_promo,
            "price_col": price_col,
            "demand_col": demand_col,
            "store_col": store_col,
            "upc_col": upc_col,
            "promo_col": promo_col,
        }