import numpy as np
import pandas as pd


class CompetitiveFeatureGenerator:
    """
    Generates cross-product competitive features for demand and elasticity estimation.

    All features use groupby.transform (no self-joins), so memory scales linearly
    with the number of rows, not quadratically with group size.

    Required columns:
        store_code, week_id, upc_code, category_code

    Optional columns (enrich neighbor definitions):
        brand_family_norm, log_price_per_liter, on_promo, units_sold,
        lag_1_log_liters_sold, miss_lag_1,
        rolling_mean_4_log_liters_sold, miss_roll_4,
        weeks_since_first_seen_store_upc

    Outputs added by run():

        # --- Group 1: Promo competitive context (clean) ---
        - n_neighbors_sw_cat
        - neighbor_promo_share_sw_cat
        - n_same_brand_neighbors_sw_cat
        - same_brand_neighbor_promo_share_sw_cat
        - n_other_brand_neighbors_sw_cat
        - other_brand_neighbor_promo_share_sw_cat

        # --- Group 2: Competitive lagged demand (clean) ---
        - n_neighbors_with_lag1
        - lag1_neighbor_mean_log_liters_sold
        - n_same_brand_neighbors_with_lag1
        - lag1_same_brand_neighbor_mean_log_liters_sold
        - n_neighbors_with_roll4
        - roll4_neighbor_mean_log_liters_sold
        - n_same_brand_neighbors_with_roll4
        - roll4_same_brand_neighbor_mean_log_liters_sold

        # --- Group 3: Static competitive structure (clean) ---
        - store_category_upc_count_static
        - store_category_brand_count_static
        - same_brand_upc_count_store_cat_static
        - share_same_brand_neighbors_static

        # --- Group 4: Competitive lifecycle (clean) ---
        - neighbor_mean_weeks_since_first_seen_store_upc
        - share_new_neighbors_13w

    Public API:
        run(df) -> pd.DataFrame
    """


    def __init__(self):
        self.group = ["store_code", "week_id", "category_code"]
        self.group_static = ["store_code", "category_code"]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- Group 1: Competitive promo context ---
        df = self._add_promo_context(df)

        # --- Group 2: Competitive lagged demand ---
        df = self._add_lagged_demand_context(df)

        # --- Group 3: Static competitive structure ---
        df = self._add_static_structure(df)

        # --- Group 4: Competitive lifecycle ---
        df = self._add_lifecycle_context(df)

        # --- Group 5: Competitive structure ---
        df = self._add_competitive_structure(df)

        return df

    # ── Group 1: Competitive promo context ───────────────────────────────────

    def _add_promo_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Promo share of neighbors in (store, week, category), split by brand.
        Uses only on_promo (no contemporaneous demand or price).
        Denominators are also stored so the model can distinguish
        'low share' from 'few neighbors'.
        """
        REQUIRED = self.group + ["upc_code", "on_promo", "brand_family_norm"]
        OUTPUT_COLS = [
            "n_neighbors_sw_cat", "neighbor_promo_share_sw_cat",
            "n_same_brand_neighbors_sw_cat", "same_brand_neighbor_promo_share_sw_cat",
        ]
        if not all(c in df.columns for c in REQUIRED):
            for col in OUTPUT_COLS:
                df[col] = np.nan
            return df

        promo_num = pd.to_numeric(df["on_promo"], errors="coerce")
        df["_promo"] = promo_num.fillna(0).astype("int8")

        # Full neighborhood (excluding self)
        group_size      = df.groupby(self.group, observed=True)["upc_code"].transform("count")
        promo_sum       = df.groupby(self.group, observed=True)["_promo"].transform("sum")
        n_neighbors     = (group_size - 1).clip(lower=0)
        neighbor_promos = (promo_sum - df["_promo"]).clip(lower=0)

        df["n_neighbors_sw_cat"] = n_neighbors
        df["neighbor_promo_share_sw_cat"] = (
            neighbor_promos.where(n_neighbors > 0) / n_neighbors.replace(0, np.nan)
        ).fillna(0)

        # Same-brand neighborhood (excluding self)
        GROUP_BRAND  = self.group + ["brand_family_norm"]
        brand_size   = df.groupby(GROUP_BRAND, observed=True)["upc_code"].transform("count")
        brand_promo  = df.groupby(GROUP_BRAND, observed=True)["_promo"].transform("sum")
        n_same       = (brand_size - 1).clip(lower=0)
        same_promos  = (brand_promo - df["_promo"]).clip(lower=0)

        df["n_same_brand_neighbors_sw_cat"] = n_same
        df["same_brand_neighbor_promo_share_sw_cat"] = (
            same_promos.where(n_same > 0) / n_same.replace(0, np.nan)
        ).fillna(0)

        df.drop(columns=["_promo"], inplace=True)
        return df

    # ── Group 2: Competitive lagged demand ───────────────────────────────────

    def _add_lagged_demand_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean lagged demand of neighbors in (store, week, category).
        Uses lag_1_log_liters_sold and rolling_mean_4_log_liters_sold
        (already computed by TemporalFeatureBuilder, never contemporaneous).
        Rows with miss_lag_1==1 or miss_roll_4==1 are excluded from the mean.
        Denominators are stored separately to distinguish 'low value' from 'no data'.
        """
        REQUIRED = self.group + ["upc_code", "brand_family_norm"]
        OUTPUT_COLS = [
             "lag1_neighbor_mean_log_liters_sold",
             "roll4_neighbor_mean_log_liters_sold",
             "lag1_same_brand_neighbor_mean_log_liters_sold",
        ]
        if not all(c in df.columns for c in REQUIRED):
            for col in OUTPUT_COLS:
                df[col] = np.nan
            return df

        GROUP_BRAND = self.group + ["brand_family_norm"]

        def _neighbor_mean_lagged(df, val_col, miss_col, mean_col, mean_brand_col):
            if val_col not in df.columns or miss_col not in df.columns:
                df[mean_col]       = np.nan
                df[mean_brand_col] = np.nan
                return df
            df["_v"] = df[val_col].where(df[miss_col] == 0)
            df["_h"] = (df[miss_col] == 0).astype(float)

            # Full neighborhood mean (excluding self)
            g_sum  = df.groupby(self.group, observed=True)["_v"].transform("sum")
            g_has  = df.groupby(self.group, observed=True)["_h"].transform("sum")
            self_v = df["_v"].fillna(0)
            n_cnt  = (g_has - df["_h"]).clip(lower=0)
            df[mean_col] = (g_sum - self_v * df["_h"]) / n_cnt.replace(0, np.nan)

            # Same-brand mean (excluding self)
            if mean_brand_col is not None:
                b_sum  = df.groupby(GROUP_BRAND, observed=True)["_v"].transform("sum")
                b_has  = df.groupby(GROUP_BRAND, observed=True)["_h"].transform("sum")
                b_cnt  = (b_has - df["_h"]).clip(lower=0)
                df[mean_brand_col] = (b_sum - self_v * df["_h"]) / b_cnt.replace(0, np.nan)

            df.drop(columns=["_v", "_h"], inplace=True)
            return df

        df = _neighbor_mean_lagged(
            df,
            val_col="lag_1_log_liters_sold", miss_col="miss_lag_1",
            mean_col="lag1_neighbor_mean_log_liters_sold",
            mean_brand_col="lag1_same_brand_neighbor_mean_log_liters_sold",
        )
        df = _neighbor_mean_lagged(
            df,
            val_col="rolling_mean_4_log_liters_sold", miss_col="miss_roll_4",
            mean_col="roll4_neighbor_mean_log_liters_sold",
            mean_brand_col=None,
        )
        return df

    # ── Group 3: Static competitive structure ────────────────────────────────

    def _add_static_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Surtido structure per (store, category) computed over the full final sample.
        Call this on df_real (after filtering), not on the raw panel, so that
        the counts reflect only the modelled universe.
        No contemporaneous price or demand used.
        """
        REQUIRED = self.group_static + ["upc_code", "brand_family_norm"]
        OUTPUT_COLS = [
            "store_category_upc_count_static",
        ]
        if not all(c in df.columns for c in REQUIRED):
            for col in OUTPUT_COLS:
                df[col] = np.nan
            return df

        upc_count = (
            df.groupby(self.group_static, observed=True)["upc_code"]
            .transform("nunique")
        )
        df["store_category_upc_count_static"] = upc_count

        return df

    # ── Group 4: Competitive lifecycle ───────────────────────────────────────

    def _add_lifecycle_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lifecycle of contemporary neighbors in (store, week, category).
        Uses weeks_since_first_seen_store_upc (already built by TemporalFeatureBuilder).
        A neighbor is considered 'new' if it has been in the store ≤ 13 weeks.
        """
        REQUIRED = self.group + ["upc_code", "weeks_since_first_seen_store_upc"]
        OUTPUT_COLS = [
            "n_new_neighbors_13w",
            "share_new_neighbors_13w",
        ]
        if not all(c in df.columns for c in REQUIRED):
            for col in OUTPUT_COLS:
                df[col] = np.nan
            return df

        w = df["weeks_since_first_seen_store_upc"]

        # Share of neighbors that are new in the store (≤ 13 weeks)
        df["_is_new"] = (w <= 13).astype(int)

        new_sum       = df.groupby(self.group, observed=True)["_is_new"].transform("sum")
        group_size    = df.groupby(self.group, observed=True)["upc_code"].transform("count")

        n_neighbors   = (group_size - 1).clip(lower=0)
        new_neighbors = (new_sum - df["_is_new"]).clip(lower=0)

        df["n_new_neighbors_13w"] = new_neighbors
        df["share_new_neighbors_13w"] = (
            new_neighbors.where(n_neighbors > 0) / n_neighbors.replace(0, np.nan)
        ).fillna(0)
        
        df.drop(columns=["_is_new"], inplace=True)
        return df

    # ── Group 5: Competitive structure ───────────────────────────────────────
    def _add_competitive_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Competitive structure per (store, category) using the modeled universe.
        """
        REQUIRED = self.group_static + ["upc_code", "brand_family_norm"]
        OUTPUT_COLS = [
            "same_brand_upc_count_store_cat_static",
        ]
        if not all(c in df.columns for c in REQUIRED):
            for col in OUTPUT_COLS:
                df[col] = np.nan
            return df

        # Distinct UPCs in each (store, category, brand)
        own_brand_upc_count = (
            df.groupby(self.group_static + ["brand_family_norm"], observed=True)["upc_code"]
            .transform("nunique")
        )
        # Exclude self UPC
        df["same_brand_upc_count_store_cat_static"] = (own_brand_upc_count - 1).clip(lower=0)

        return df
