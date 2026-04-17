import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union

from .functions.missing_data import NanAnalyzer
from .functions.outliers import OutlierAnalyzer
from .functions.competitors import CompetitiveFeatureGenerator
from .functions.time_series import (
    AutocorrelationAnalyzer,
    AggregatedTrendAnalyzer,
    TemporalFeatureBuilder,
)
from .functions.grain import (
    GrainUniquenessAnalyzer,
    DatasetCoverageAnalyzer,
    CalendarGapImputer,
    PanelBalanceAnalyzer,
    StoreWeekCoverageAnalyzer,
    StoreWeekWidthAnalyzer,
    StoreUpcCoverageAnalyzer,
)
from .functions.price_variation import (
    PriceVariationAnalyzer,
    PromoPriceCollinearityAnalyzer,
    LogPriceLogDemandAnalyzer,
    BaselineElasticityOLSAnalyzer,
)


class EDA:
    """
    Facade for exploratory data analysis of Dominick's panel data.

    Provides a unified interface over 13 specialized analyzers, grouped into
    four analysis areas:

      Data Quality:
        analyze_nans, analyze_grain_uniqueness, analyze_dataset_coverage,
        impute_calendar_gaps, analyze_panel_balance, analyze_store_week_coverage,
        analyze_store_week_width, analyze_store_upc_coverage

      Price Variation:
        analyze_price_variation, analyze_promo_price_collinearity,
        analyze_log_price_log_demand, analyze_baseline_elasticity_ols

      Outliers:
        analyze_outliers

      Time Series:
        analyze_aggregated_trends, build_temporal_features, analyze_autocorrelation

    Each method delegates directly to its analyzer's .run() method.
    All analyzers are instantiated internally with default configuration.
    """

    def __init__(self):
        self.nan_analyzer = NanAnalyzer()

        self.aggregated_trend_analyzer = AggregatedTrendAnalyzer()
        self.temporal_feature_builder = TemporalFeatureBuilder()

        self.grain_uniqueness_analyzer = GrainUniquenessAnalyzer()
        self.dataset_coverage_analyzer = DatasetCoverageAnalyzer()
        self.calendar_gap_imputer = CalendarGapImputer()
        self.panel_balance_analyzer = PanelBalanceAnalyzer()
        self.store_week_coverage_analyzer = StoreWeekCoverageAnalyzer()
        self.store_week_width_analyzer = StoreWeekWidthAnalyzer()
        self.store_upc_coverage_analyzer = StoreUpcCoverageAnalyzer()
        self.price_variation_analyzer = PriceVariationAnalyzer()
        self.promo_price_collinearity_analyzer = PromoPriceCollinearityAnalyzer()
        self.log_price_log_demand_analyzer = LogPriceLogDemandAnalyzer()
        self.baseline_elasticity_ols_analyzer = BaselineElasticityOLSAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer()
        self.autocorrelation_analyzer = AutocorrelationAnalyzer()

        self.competitive_feature_generator = CompetitiveFeatureGenerator()

### ---------------------- Data Quality ----------------------
    def analyze_nans(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        return self.nan_analyzer.run(df, columns=columns)

    def analyze_grain_uniqueness(self, df: pd.DataFrame, grain_cols: Optional[list[str]] = None) -> pd.DataFrame:
        if grain_cols is None:
            grain_cols = ["store_code", "week_id", "upc_code"]
        return self.grain_uniqueness_analyzer.run(df, grain_cols)

    def analyze_dataset_coverage(self, df: pd.DataFrame, store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id", expected_weeks: Optional[list[int]] = None) -> dict:
        return self.dataset_coverage_analyzer.run(df, store_col, upc_col, week_col, expected_weeks)

    def impute_calendar_gaps(self, df: pd.DataFrame, store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id", value_cols: Optional[list[str]] = None, global_gap_weeks: Optional[list[int]] = None) -> pd.DataFrame:
        return self.calendar_gap_imputer.run(df, store_col, upc_col, week_col, value_cols, global_gap_weeks)

    def analyze_panel_balance(self, df: pd.DataFrame, store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id") -> dict:
        return self.panel_balance_analyzer.run(df, store_col, upc_col, week_col)

    def analyze_store_week_coverage(self, df: pd.DataFrame, store_col: str = "store_code", week_col: str = "week_id", min_weeks_for_good: int = 150) -> dict:
        return self.store_week_coverage_analyzer.run(df, store_col, week_col, min_weeks_for_good)

    def analyze_store_week_width(
        self, df: pd.DataFrame, store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id") -> dict:
        return self.store_week_width_analyzer.run(df, store_col, upc_col, week_col)

    def analyze_store_upc_coverage(
        self, df: pd.DataFrame, store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id") -> dict:
        return self.store_upc_coverage_analyzer.run(df, store_col, upc_col, week_col)

### ---------------------- Price Variation ----------------------
    def analyze_price_variation(self, df: pd.DataFrame, price_col: str = "log_price_per_liter", store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id", price_round_decimals: int = 3) -> dict:
        return self.price_variation_analyzer.run(df, price_col, store_col, upc_col, week_col, price_round_decimals)
    
    def analyze_promo_price_collinearity(self, df: pd.DataFrame, price_col: str = "log_price_per_liter", store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id", on_promo_col: str = "on_promo", promo_b_col: str = "promo_B", promo_s_col: str = "promo_S", promo_c_col: str = "promo_C", price_change_tol: float = 1e-6) -> dict:
        return self.promo_price_collinearity_analyzer.run(df, price_col, store_col, upc_col, week_col, on_promo_col, promo_b_col, promo_s_col, promo_c_col, price_change_tol)

    def analyze_log_price_log_demand(self, df: pd.DataFrame, price_col: str = "log_price_per_liter", demand_col: str = "log_liters_sold", store_col: str = "store_code", upc_col: str = "upc_code", promo_col: str = "on_promo") -> dict:
        return self.log_price_log_demand_analyzer.run(df, price_col, demand_col, store_col, upc_col, promo_col)

    def analyze_baseline_elasticity_ols(self, df: pd.DataFrame, price_col: str = "log_price_per_liter", demand_col: str = "log_liters_sold", store_col: str = "store_code", upc_col: str = "upc_code", week_col: str = "week_id") -> dict:
        return self.baseline_elasticity_ols_analyzer.run(df, price_col, demand_col, store_col, upc_col, week_col)

### --- Outliers Analysis ---
    def analyze_outliers(self, df: pd.DataFrame, columns: list[str], method: str = "std", n_std: float = 3.0, group_col: Optional[Union[str, list[str]]] = None, return_flagged_df: bool = False) -> Union[pd.DataFrame, dict]:
        return self.outlier_analyzer.run(df, columns, method=method, n_std=n_std, group_col=group_col, return_flagged_df=return_flagged_df)

### ---------------------- Time Series Analysis ----------------------
    def analyze_aggregated_trends(self, df: pd.DataFrame, time_col: str = "week_id", price_col: str = "log_price_per_liter", demand_raw_col: str = "liters_sold", promo_col: str = "on_promo") -> dict:
        return self.aggregated_trend_analyzer.run(df, time_col=time_col, price_col=price_col, demand_raw_col=demand_raw_col, promo_col=promo_col)

    def build_temporal_features(
        self, df: pd.DataFrame, week_col: str = "week_id", store_col: str = "store_code", upc_col: str = "upc_code", demand_col: str = "log_liters_sold", promo_col: str = "on_promo", season_periods: Optional[list[int]] = None, lag_weeks: Optional[list[int]] = None, rolling_windows: Optional[list[int]] = None, include_lifecycle_upc: bool = True, include_lifecycle_store_upc: bool = True, include_promo_intensity: bool = True) -> pd.DataFrame:
        return self.temporal_feature_builder.run(df, week_col, store_col, upc_col, demand_col, promo_col, season_periods, lag_weeks, rolling_windows, include_lifecycle_upc, include_lifecycle_store_upc, include_promo_intensity)

    def analyze_autocorrelation(self, df: pd.DataFrame, value_col: str = "log_liters_sold", max_lags: int = 20) -> dict:
        return self.autocorrelation_analyzer.run(df, value_col, max_lags)

### ---------------------- Competitive Features ----------------------
    def build_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.competitive_feature_generator.run(df)