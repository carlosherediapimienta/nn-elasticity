from dataclasses import dataclass, field
from typing import List


@dataclass
class ElasticityConfig:
    csv_path: str
    target_col: str = "log_liters_sold"
    price_col: str = "log_price_per_liter"

    numeric_control_cols: List[str] = field(default_factory=lambda: [
        "on_promo",
        "week_rank",
        "sin_52",
        "cos_52",
        "sin_26",
        "cos_26",
        "sin_13",
        "cos_13",
        "weeks_since_first_seen_upc",
        "weeks_since_first_seen_store_upc",
        "liters_per_upc",
        "lag_1_log_liters_sold",
        "lag_2_log_liters_sold",
        "lag_4_log_liters_sold",
        "rolling_mean_4_log_liters_sold",
        "rolling_mean_8_log_liters_sold",
        "rolling_mean_13_log_liters_sold",
        "miss_lag_1",
        "miss_lag_2",
        "miss_lag_4",
        "miss_roll_4",
        "miss_roll_8",
        "miss_roll_13",
        "promo_intensity_store_week",
    ])

    store_col: str = "store_code"
    upc_col: str = "upc_code"
    category_col: str = "category_code"

    include_store_fixed_effect: bool = True
    include_upc_fixed_effect: bool = True
    include_category_fixed_effect: bool = False

    robust_cov_type: str = "HC1"
    min_obs: int = 30