import pandas as pd
from typing import List

class PairDatasetBuilder:
    """
    Transforms a long-format panel into a directed pairwise dataset.
    For each (store, week), produces one row per ordered pair (upc_i, upc_j)
    carrying upc_i's own demand and log-price alongside upc_j's log-price.
    A canonical ``pair_id`` (sorted, so i<->j share the same id) is added to
    allow downstream symmetrization of cross-elasticities.
    """
    # Base columns to keep
    BASE_COLS = ["store_code", "week_id", "upc_code", "log_liters_sold", "log_price_per_liter"]

    def __init__(self, control_cols: List[str]) -> None:
        self.control_cols = control_cols

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        # Base columns + control columns
        pair_base = df[self.BASE_COLS + self.control_cols].copy()

        # Left side of the pair: upc_i's own demand and log-price
        left = pair_base.rename(columns={
            "upc_code": "upc_i",
            "log_liters_sold": "log_v_i",
            "log_price_per_liter": "log_p_i",
        })

        # Right side of the pair: upc_j's log-price
        right = pair_base[["store_code", "week_id", "upc_code", "log_price_per_liter"]].rename(
            columns={
                "upc_code": "upc_j",
                "log_price_per_liter": "log_p_j",
            }
        )

        # Merge the left and right sides of the pair: upc_i's own demand and log-price alongside upc_j's log-price
        pair_df = left.merge(right, on=["store_code", "week_id"], how="inner")
        pair_df = pair_df[pair_df["upc_i"] != pair_df["upc_j"]].copy()

        # Create the pair_id: canonical pair_id (sorted, so i<->j share the same id)
        pair_df["pair_id"] = pair_df.apply(
            lambda r: "__".join(map(str, sorted([r["upc_i"], r["upc_j"]]))),
            axis=1,
        )

        return pair_df