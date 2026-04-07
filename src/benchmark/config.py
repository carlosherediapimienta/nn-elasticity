from dataclasses import dataclass, field
from typing import List

@dataclass
class BenchmarkConfig:
    """Shared configuration for the pairwise OLS benchmark.
    Attributes:
        control_cols: RHS regressors added to every OLS fit beyond own/cross log-prices.
        min_obs: Minimum number of clean training rows required to attempt a fit.
        robust_cov_type: Heteroskedasticity-robust covariance estimator passed to
            statsmodels (e.g. "HC1", "HC3").
    """
    # Control columns
    control_cols: List[str] = field(default_factory=lambda: [
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

    # Model parameters
    min_obs: int = 30
    # Robust covariance type. The OLS method assumes homoskedasticity (constant variance),
    # and this is not always the case because we can have variability in each store-week. 
    # So we use a robust estimator to account for heteroskedasticity.
    robust_cov_type: str = "HC1"