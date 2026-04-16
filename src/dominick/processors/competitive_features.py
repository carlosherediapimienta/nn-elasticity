import numpy as np
import pandas as pd


class CompetitiveFeatureGenerator:
    """
    Generates cross-product competitive features for demand and elasticity estimation.

    All features use groupby.transform (no self-joins), so memory scales linearly
    with the number of rows, not quadratically with group size.

    Required columns:
        store_code, week_id, upc_code, category_code, units_sold

    Optional columns (enrich neighbor definitions):
        brand_family_norm, log_price_per_liter, on_promo

    Outputs added by run():
        - n_active_neighbors_sw_cat
        - neighbor_weighted_mean_log_price
        - neighbor_min_log_price
        - neighbor_promo_share
        - price_gap_to_neighbors_mean
        - price_gap_to_cheapest_neighbor

    Public API:
        run(df) -> pd.DataFrame
    """

    GROUP = ["store_code", "week_id", "category_code"]
    
    def __init__(self):
        self.active_min_units = 0

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_n_active_neighbors_sw_cat(df)
        df = self._add_neighbor_weighted_mean_log_price(df)
        df = self._add_neighbor_min_log_price(df)
        df = self._add_neighbor_promo_share(df)
        df = self._add_price_gap_to_neighbors_mean(df)
        df = self._add_price_gap_to_cheapest_neighbor(df)
        return df

    # ── Shared infrastructure ────────────────────────────────────────────────

    def _has_cols(self, df: pd.DataFrame, required: list[str], output_col: str) -> bool:
        if not all(c in df.columns for c in required):
            df[output_col] = np.nan
            return False
        return True

    def _is_active(self, df: pd.DataFrame) -> pd.Series:
        return (df["units_sold"].fillna(0) > self.active_min_units).astype(int)

    def _add_price_gap(self, df: pd.DataFrame, neighbor_col: str, output_col: str) -> pd.DataFrame:
        if not self._has_cols(df, ["log_price_per_liter", neighbor_col], output_col):
            return df
        df[output_col] = df["log_price_per_liter"] - df[neighbor_col]
        return df

    # ── Feature methods ──────────────────────────────────────────────────────

    def _add_n_active_neighbors_sw_cat(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Number of active competing SKUs in the same (store, week, category).
        Self excluded. Formula: active_in_group - is_active_self.
        Uses groupby.transform("sum").
        """
        if not self._has_cols(df, self.GROUP + ["upc_code", "units_sold"], "n_active_neighbors_sw_cat"):
            return df

        df["_is_active"] = self._is_active(df)
        active_in_group  = df.groupby(self.GROUP, observed=True)["_is_active"].transform("sum")
        df["n_active_neighbors_sw_cat"] = (active_in_group - df["_is_active"]).clip(lower=0).astype(int)
        df.drop(columns=["_is_active"], inplace=True)
        return df

    def _add_neighbor_weighted_mean_log_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted mean of competitors' log-price, excluding self.
        Weight: ×2 for same brand, ×1 for different brand.

        Uses group-level sums and brand-level sums:
            numerator_i   = (group_sum + brand_sum) - 2 * p_i  (if p_i valid)
            denominator_i = (group_count + brand_count) - 2    (if p_i valid)

        Derivation:
            ∑_{j≠i} w_ij * p_j  =  2*(brand_sum - p_i)  +  1*(group_sum - brand_sum)
                                 =  group_sum + brand_sum - 2*p_i
            ∑_{j≠i} w_ij        =  2*(brand_count - 1)  +  (group_count - brand_count)
                                 =  group_count + brand_count - 2
        """
        REQUIRED = self.GROUP + ["upc_code", "log_price_per_liter", "brand_family_norm"]

        if not self._has_cols(df, REQUIRED, "neighbor_weighted_mean_log_price"):
            return df

        g_sum   = df.groupby(self.GROUP, observed=True)["log_price_per_liter"].transform("sum")
        g_count = df.groupby(self.GROUP, observed=True)["log_price_per_liter"].transform("count")

        b_sum   = df.groupby(self.GROUP + ["brand_family_norm"], observed=True)["log_price_per_liter"].transform("sum")
        b_count = df.groupby(self.GROUP + ["brand_family_norm"], observed=True)["log_price_per_liter"].transform("count")

        has_price = df["log_price_per_liter"].notna().astype(int)
        p_i       = df["log_price_per_liter"].fillna(0)

        numerator   = (g_sum + b_sum) - 2 * p_i * has_price
        denominator = (g_count + b_count) - 2 * has_price

        df["neighbor_weighted_mean_log_price"] = numerator / denominator.replace(0, np.nan)
        return df

    def _add_neighbor_min_log_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimum log-price among active competitors, excluding self.

        Uses group min and group second-min:
            - If self is NOT the unique minimum  → neighbor_min = group_min
            - If self IS the unique minimum       → neighbor_min = group_second_min
            - Ties at the minimum are safe        → group_min is still valid (others share it)
        Only active products (units_sold > 0) are considered.
        """
        REQUIRED = self.GROUP + ["upc_code", "log_price_per_liter", "units_sold"]

        if not self._has_cols(df, REQUIRED, "neighbor_min_log_price"):
            return df

        # Mask inactive products so they don't affect the min
        df["_p_active"] = df["log_price_per_liter"].where(self._is_active(df) == 1)

        group_min = df.groupby(self.GROUP, observed=True)["_p_active"].transform("min")

        # Second min: minimum of active prices strictly above group_min
        def _second_min(x: pd.Series) -> pd.Series:
            m = x.min()
            above = x[x > m]
            s = above.min() if not above.empty else np.nan
            return pd.Series(s, index=x.index)

        group_min2 = df.groupby(self.GROUP, observed=True)["_p_active"].transform(_second_min)

        # Count how many active products share the group minimum
        df["_at_min"]    = (df["_p_active"] == group_min).astype("Int8")
        count_at_min     = df.groupby(self.GROUP, observed=True)["_at_min"].transform("sum")

        # Use second_min only when self is the unique holder of the group minimum
        is_unique_min = (df["_p_active"] == group_min) & (count_at_min == 1)
        df["neighbor_min_log_price"] = np.where(is_unique_min, group_min2, group_min)

        df.drop(columns=["_p_active", "_at_min"], inplace=True)
        return df

    def _add_neighbor_promo_share(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fraction of competitors on promo in the same (store, week, category).
        Formula: #{j≠i : on_promo_j=1} / #{j≠i}. NaN for singleton groups.
        """
        if not self._has_cols(df, self.GROUP + ["upc_code", "on_promo"], "neighbor_promo_share"):
            return df

        df["_on_promo"]  = df["on_promo"].fillna(0).astype(int)
        promo_in_group   = df.groupby(self.GROUP, observed=True)["_on_promo"].transform("sum")
        group_size       = df.groupby(self.GROUP, observed=True)["upc_code"].transform("count")
        n_neighbors      = (group_size - 1).clip(lower=0)
        neighbor_promos  = (promo_in_group - df["_on_promo"]).clip(lower=0)

        df["neighbor_promo_share"] = (
            neighbor_promos
            .where(n_neighbors > 0, other=np.nan)
            / n_neighbors.replace(0, np.nan)
        )
        df.drop(columns=["_on_promo"], inplace=True)
        return df

    def _add_price_gap_to_neighbors_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """log_price_i - neighbor_weighted_mean_log_price. Requires run order."""
        return self._add_price_gap(df, "neighbor_weighted_mean_log_price", "price_gap_to_neighbors_mean")

    def _add_price_gap_to_cheapest_neighbor(self, df: pd.DataFrame) -> pd.DataFrame:
        """log_price_i - neighbor_min_log_price. Requires run order."""
        return self._add_price_gap(df, "neighbor_min_log_price", "price_gap_to_cheapest_neighbor")