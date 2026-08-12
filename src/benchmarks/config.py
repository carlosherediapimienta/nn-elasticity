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
        "sin_13",
        "cos_13",
        "weeks_since_first_seen_store_upc",
        "lag_1_log_liters_sold",
        "lag_4_log_liters_sold",
        "miss_lag_1",
        "miss_lag_4",
        "promo_intensity_store_week",
        "n_neighbors_sw_cat",
        "neighbor_promo_share_sw_cat",
        "lag1_neighbor_mean_log_liters_sold",
        "share_new_neighbors_13w",
    ])

    # Model parameters
    min_obs: int = 30
    # Robust covariance type. The OLS method assumes homoskedasticity (constant variance),
    # and this is not always the case because we can have variability in each store-week. 
    # So we use a robust estimator to account for heteroskedasticity.
    robust_cov_type: str = "HC1"

@dataclass
class RidgeConfig:
    """
    Regularized log-log demand benchmark config. Mirrors BenchmarkConfig's
    grouping (per store, per pair) -- only the estimator changes (Ridge vs
    OLS), isolating the effect of coefficient shrinkage.
    """
    control_cols: List[str] = field(default_factory=lambda: BenchmarkConfig().control_cols)
    alphas: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 30.0, 100.0, 300.0])
    cv_folds: int = 5
    min_obs: int = 30
    standardize: bool = True


@dataclass
class MLPConfig:
    """
    Generic demand-first MLP config. Deliberately smaller than ICDN's
    encoder (256, 128, 64) and without splines, attention, U_ij
    interaction, or elasticity penalties.
    """
    control_cols: List[str] = field(default_factory=lambda: BenchmarkConfig().control_cols)
    hidden: tuple = (64, 32)
    act: str = "gelu"
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    n_epochs: int = 200
    es_patience: int = 25
    huber_delta: float = 1.0