import pandas as pd
import numpy as np
from typing import Optional


class TemporalFeatureBuilder:
    """
    Adds temporal features oriented to elasticity:
    week_rank, seasonal sin/cos, weeks_since_first_seen, lags/rolling demand,
    promo_intensity by store-week.
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        week_col: str = "week_id",
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        demand_col: str = "log_liters_sold",
        promo_col: str = "on_promo",
        season_periods: Optional[list[int]] = None,
        lag_weeks: Optional[list[int]] = None,
        rolling_windows: Optional[list[int]] = None,
        include_lifecycle_upc: bool = True,
        include_lifecycle_store_upc: bool = True,
        include_promo_intensity: bool = True,
    ) -> pd.DataFrame:
        """
        Args:
            df                       : DataFrame with grain (store, upc, week).
                                       Sorted internally if not already.
            week_col                 : numeric week identifier column.
            store_col                : store identifier column.
            upc_col                  : product identifier column.
            demand_col               : demand column used for lags and rolling stats
                                       (default: log_liters_sold).
            promo_col                : binary promo column (0/1) used for promo_intensity
                                       (default: on_promo).
            season_periods           : periods for sin/cos seasonality encoding
                                       (default: [52]; consider also [52, 26, 13]).
            lag_weeks                : demand lag periods to add (default: [1, 2, 4]).
            rolling_windows          : windows for rolling mean/median (default: [4, 8, 13]).
            include_lifecycle_upc    : add weeks_since_first_seen per UPC.
            include_lifecycle_store_upc: add weeks_since_first_seen per store × UPC.
            include_promo_intensity  : add share of UPCs on promo per store-week.

        Returns:
            Original DataFrame with new columns appended (input is not modified).
        """
        if season_periods is None:
            season_periods = [52]
        if lag_weeks is None:
            lag_weeks = [1, 2, 4]
        if rolling_windows is None:
            rolling_windows = [4, 8, 13]

        out = df.copy()
        for c in [week_col, store_col, upc_col]:
            if c not in out.columns:
                raise ValueError(f"Column '{c}' not found.")

        # ----- week_rank (sequential index without gaps) -----
        unique_weeks = np.sort(out[week_col].dropna().unique())
        week_to_rank = {w: i + 1 for i, w in enumerate(unique_weeks)}
        out["week_rank"] = out[week_col].map(week_to_rank)

        # ----- Seasonality sin/cos -----
        for p in season_periods:
            out[f"sin_{p}"] = np.sin(2 * np.pi * out["week_rank"] / p)
            out[f"cos_{p}"] = np.cos(2 * np.pi * out["week_rank"] / p)

        # ----- Lifecycle: weeks_since_first_seen -----
        if include_lifecycle_upc:
            first_rank_upc = out.groupby(upc_col)["week_rank"].transform("min")
            out["weeks_since_first_seen_upc"] = (out["week_rank"] - first_rank_upc).astype(int)
        if include_lifecycle_store_upc:
            first_rank_su = out.groupby([store_col, upc_col])["week_rank"].transform("min")
            out["weeks_since_first_seen_store_upc"] = (out["week_rank"] - first_rank_su).astype(int)

        # ----- Lags and rolling (within store×upc, ordered by week) -----
        if demand_col in out.columns:
                out = out.sort_values([store_col, upc_col, week_col]).reset_index(drop=True)
                g = out.groupby([store_col, upc_col])[demand_col]
                for k in lag_weeks:
                    col_lag = f"lag_{k}_{demand_col}"
                    out[col_lag] = g.shift(k)
                    out[f"miss_lag_{k}"] = (out[col_lag].isna() | ~np.isfinite(out[col_lag])).astype(int)
                    out[col_lag] = out[col_lag].fillna(0.0)
                # Rolling strictly historical: mean/median of the w weeks BEFORE (excluding t)
                for w in rolling_windows:
                    col_rm = f"rolling_mean_{w}_{demand_col}"
                    col_rmed = f"rolling_median_{w}_{demand_col}"
                    out[col_rm] = g.transform(
                        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                    )
                    out[col_rmed] = g.transform(
                        lambda x: x.shift(1).rolling(w, min_periods=1).median()
                    )
                    out[f"miss_roll_{w}"] = (
                        (out[col_rm].isna() | ~np.isfinite(out[col_rm]))
                        | (out[col_rmed].isna() | ~np.isfinite(out[col_rmed]))
                    ).astype(int)
                    out[col_rm] = out[col_rm].fillna(0.0)
                    out[col_rmed] = out[col_rmed].fillna(0.0)

        # ----- Promo intensity by store-week (share of UPCs on promo) -----
        if include_promo_intensity and promo_col in out.columns:
            store_week_promo = (
                out.groupby([store_col, week_col])[promo_col]
                .mean()
                .reset_index()
                .rename(columns={promo_col: "promo_intensity_store_week"})
            )
            out = out.merge(store_week_promo, on=[store_col, week_col], how="left")

        return out