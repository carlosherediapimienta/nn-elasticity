import pandas as pd
import numpy as np


class PromoPriceCollinearityAnalyzer:
    """
    Analyzes the relationship between price and promo:
    - Prevalence and consistency of promo flags
    - Mean price difference between promo and no promo
    - Collinearity of log_price vs on_promo within (store, upc)
    - Proportion of price changes linked to promo changes
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        price_col: str = "log_price_per_liter",
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        week_col: str = "week_id",
        on_promo_col: str = "on_promo",
        promo_b_col: str = "promo_B",
        promo_s_col: str = "promo_S",
        promo_c_col: str = "promo_C",
        price_change_tol: float = 1e-6,
    ) -> dict:
        """
        Args:
            df: DataFrame with grain (store, upc, week) and price/promo columns.
            price_col: price column in log.
            store_col, upc_col, week_col: grain columns.
            on_promo_col: global promo flag (0/1).
            promo_*_col: specific promo type flags (0/1).
            price_change_tol: absolute tolerance to consider that the price changes.

        Returns:
            dict with:
                - global_stats: aggregated metrics about promos and discounts.
                - per_series: DataFrame by (store, upc) with:
                    * store_col, upc_col
                    * n_obs: number of observations
                    * n_obs_promo, n_obs_no_promo: number of observations with promo and no promo
                    * has_both_promo_states: bool, True if the series has both promo states
                    * mean_price_promo, mean_price_no_promo, price_promo_gap: mean price with promo and no promo, price difference between promo and no promo
                    * corr_price_on_promo: correlation between price and promo
                    * n_price_changes: number of price changes
                    * n_price_changes_with_promo_switch: number of price changes with promo switch
                    * share_changes_with_promo_switch: share of price changes with promo switch
        """
        required_cols = [
            price_col, store_col, upc_col, week_col,
            on_promo_col, promo_b_col, promo_s_col, promo_c_col,
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # Work on minimal copy
        df_work = df[required_cols].copy()

        # Normalize flags to 0/1 (treating NaN as 0)
        for c in [on_promo_col, promo_b_col, promo_s_col, promo_c_col]:
            df_work[c] = df_work[c].fillna(0).astype(int)

        # --- Global promo statistics ---
        n_rows = int(len(df_work))
        promo_b_share = float(df_work[promo_b_col].mean())
        promo_s_share = float(df_work[promo_s_col].mean())
        promo_c_share = float(df_work[promo_c_col].mean())

        # Consistency of on_promo and specific promo flags
        any_promo = (
            (df_work[promo_b_col] == 1)
            | (df_work[promo_s_col] == 1)
            | (df_work[promo_c_col] == 1)
        ).astype(int)
        on_promo = df_work[on_promo_col]

        on_promo_matches_any = bool(((on_promo == any_promo)).all())

        sum_promos = (
            df_work[promo_b_col] + df_work[promo_s_col] + df_work[promo_c_col]
        )
        promos_mutually_exclusive = bool((sum_promos <= 1).all())

        # --- Mean price difference between promo and no promo (by series) ---
        # Filter rows with defined price
        df_price = df[[store_col, upc_col, week_col, price_col, on_promo_col]].copy()
        df_price = df_price.dropna(subset=[price_col])
        df_price[on_promo_col] = df_price[on_promo_col].fillna(0).astype(int)

        per_series_rows = []

        for (s, u), g in df_price.groupby([store_col, upc_col]):
            g = g.sort_values(week_col) if week_col in g.columns else g
            n_obs = len(g)
            if n_obs == 0:
                continue

            g_promo = g[g[on_promo_col] == 1]
            g_no = g[g[on_promo_col] == 0]

            n_obs_promo = int(len(g_promo))
            n_obs_no_promo = int(len(g_no))
            has_both = (n_obs_promo > 0) and (n_obs_no_promo > 0)

            if has_both:
                mean_price_promo = float(g_promo[price_col].mean())
                mean_price_no_promo = float(g_no[price_col].mean())
                price_promo_gap = mean_price_promo - mean_price_no_promo
            else:
                mean_price_promo = np.nan
                mean_price_no_promo = np.nan
                price_promo_gap = np.nan

            # Correlation of price–promo within series (only if both states)
            if has_both and g[on_promo_col].nunique() > 1 and g[price_col].nunique() > 1:
                corr = float(
                    g[[price_col, on_promo_col]].corr().iloc[0, 1]
                )
            else:
                corr = np.nan

            # Price changes and if they coincide with promo changes
            g = g.sort_values(week_col)
            price_diff = g[price_col].diff().abs() > price_change_tol
            promo_switch = g[on_promo_col].diff().fillna(0) != 0

            n_price_changes = int(price_diff.sum())
            n_price_changes_with_promo_switch = int((price_diff & promo_switch).sum())

            if n_price_changes > 0:
                share_changes_with_promo_switch = (
                    n_price_changes_with_promo_switch / n_price_changes
                )
            else:
                share_changes_with_promo_switch = np.nan

            per_series_rows.append({
                store_col: s,
                upc_col: u,
                "n_obs": int(n_obs),
                "n_obs_promo": n_obs_promo,
                "n_obs_no_promo": n_obs_no_promo,
                "has_both_promo_states": has_both,
                "mean_price_promo": mean_price_promo,
                "mean_price_no_promo": mean_price_no_promo,
                "price_promo_gap": price_promo_gap,
                "corr_price_on_promo": corr,
                "n_price_changes": n_price_changes,
                "n_price_changes_with_promo_switch": n_price_changes_with_promo_switch,
                "share_changes_with_promo_switch": share_changes_with_promo_switch,
            })

        per_series = pd.DataFrame(per_series_rows)

        # Series with both promo states (for some metrics)
        mask_both = per_series["has_both_promo_states"]
        gaps_series = per_series.loc[mask_both, "price_promo_gap"].dropna()

        if gaps_series.shape[0] > 0:
            price_promo_gap_mean = float(gaps_series.mean())
            price_promo_gap_median = float(gaps_series.median())
            pct_series_promo_more_expensive = float(
                (gaps_series > 0).mean() * 100
            )
        else:
            price_promo_gap_mean = np.nan
            price_promo_gap_median = np.nan
            pct_series_promo_more_expensive = np.nan

        # Correlation of log_price – on_promo by series
        corr_series = per_series["corr_price_on_promo"].dropna()
        if corr_series.shape[0] > 0:
            corr_mean = float(corr_series.mean())
            corr_median = float(corr_series.median())
        else:
            corr_mean = np.nan
            corr_median = np.nan

        # share_changes_with_promo_switch by series
        share_series = per_series["share_changes_with_promo_switch"].dropna()
        if share_series.shape[0] > 0:
            share_p50 = float(share_series.quantile(0.50))
            share_p75 = float(share_series.quantile(0.75))
            share_p90 = float(share_series.quantile(0.90))
        else:
            share_p50 = share_p75 = share_p90 = np.nan

        global_stats = {
            "n_rows": n_rows,
            "promo_shares": {
                promo_b_col: promo_b_share,
                promo_s_col: promo_s_share,
                promo_c_col: promo_c_share,
            },
            "on_promo_matches_any_promo": on_promo_matches_any,
            "promos_mutually_exclusive": promos_mutually_exclusive,
            "price_promo_gap": {
                "mean": price_promo_gap_mean,
                "median": price_promo_gap_median,
                "pct_series_promo_more_expensive": pct_series_promo_more_expensive,
            },
            "corr_price_on_promo": {
                "mean": corr_mean,
                "median": corr_median,
            },
            "share_changes_with_promo_switch": {
                "p50": share_p50,
                "p75": share_p75,
                "p90": share_p90,
            },
            "store_col": store_col,
            "upc_col": upc_col,
            "week_col": week_col,
            "price_col": price_col,
            "on_promo_col": on_promo_col,
        }

        return {
            "global_stats": global_stats,
            "per_series": per_series,
        }