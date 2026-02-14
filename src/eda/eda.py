import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from .functions.missing_data import NanAnalyzer
from .functions.correlation import (
    CorrelationAnalyzer,
    GlobalCorrelationAnalyzer,
    CorrelationHistogramPlotter,
    ScatterRegressionPlotter,
)
from .functions.distribution import (
    DistributionAnalyzer,
    DistributionPlotter,
)
from .functions.outliers import (
    OutlierAnalyzer,
    BoxPlotter,
)
from .functions.time_series import (
    TimeSeriesAggregator,
    TrendAnalyzer,
    AutocorrelationAnalyzer,
    SeasonalityDetector,
    TimeSeriesPlotter,
    AutocorrelationPlotter,
)


class EDA:
    """
    Orchestrator of EDA analysis. Coordinates NanAnalyzer, CorrelationAnalyzer
    and CorrelationHistogramPlotter, DistributionAnalyzer, OutlierAnalyzer,
    GlobalCorrelationAnalyzer, ScatterRegressionPlotter, TimeSeriesAggregator,
    TrendAnalyzer, AutocorrelationAnalyzer, SeasonalityDetector, TimeSeriesPlotter,
    AutocorrelationPlotter using only their public API (run).
    Public API of this class: run().
    """

    def __init__(self):
        self.nan_analyzer = NanAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.histogram_plotter = CorrelationHistogramPlotter()
        self.distribution_analyzer = DistributionAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer()
        self.distribution_plotter = DistributionPlotter()
        self.box_plotter = BoxPlotter()
        self.global_correlation_analyzer = GlobalCorrelationAnalyzer()
        self.scatter_regression_plotter = ScatterRegressionPlotter()

        self.time_series_aggregator = TimeSeriesAggregator()
        self.trend_analyzer = TrendAnalyzer()
        self.autocorrelation_analyzer = AutocorrelationAnalyzer()
        self.seasonality_detector = SeasonalityDetector()

        self.time_series_plotter = TimeSeriesPlotter()
        self.autocorrelation_plotter = AutocorrelationPlotter()

    def analyze_nans(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        return self.nan_analyzer.run(df, columns=columns)
    
    def analyze_correlations(self, df: pd.DataFrame, x_col: str, y_col: str, group_col: str = "store_code", min_obs: int = 30) -> pd.DataFrame:
        return self.correlation_analyzer.run(df, x_col, y_col, group_col=group_col, min_obs=min_obs)
    
    def plot_correlations(self, corr_df: pd.DataFrame, title: str = "Spearman correlation by store", xlabel: str = "Spearman correlation", ylabel: str = "Number of stores", bins: int = 25, figsize: tuple[int, int] = (8, 5), ax: Optional[plt.Axes] = None) -> plt.Axes:
        return self.histogram_plotter.run(corr_df, title=title, xlabel=xlabel, ylabel=ylabel, bins=bins, figsize=figsize, ax=ax)
    
    def analyze_distributions(self, df: pd.DataFrame, columns: list[str]) -> dict:
        return self.distribution_analyzer.run(df, columns)
    
    def analyze_outliers(self, df: pd.DataFrame, columns: list[str], method: str = 'std', n_std: float = 3.0) -> pd.DataFrame:
        return self.outlier_analyzer.run(df, columns, method=method, n_std=n_std)
    
    def plot_distributions(self, df: pd.DataFrame, columns: list[str], bins: int = 50, figsize: tuple[int, int] = (14, 5), stats_dict: dict = None) -> plt.Figure:
        return self.distribution_plotter.run(df, columns, bins=bins, figsize=figsize, stats_dict=stats_dict)
    
    def plot_boxplots(self, df: pd.DataFrame, columns: list[str], figsize: tuple[int, int] = (12, 5)) -> plt.Figure:
        return self.box_plotter.run(df, columns, figsize=figsize)

    def analyze_global_correlation(self, df: pd.DataFrame, x_col: str, y_col: str) -> dict:
        return self.global_correlation_analyzer.run(df, x_col, y_col)

    def plot_scatter_regression(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        sample_size: Optional[int] = None,
        figsize: tuple[int, int] = (10, 7),
        stats_dict: dict = None,
        alpha: float = 0.3
    ) -> plt.Figure:
        return self.scatter_regression_plotter.run(
            df, x_col, y_col, 
            sample_size=sample_size, 
            figsize=figsize, 
            stats_dict=stats_dict, 
            alpha=alpha
        )

    def aggregate_time_series(
        self, 
        df: pd.DataFrame, 
        time_col: str, 
        value_cols: list[str],
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        return self.time_series_aggregator.run(df, time_col, value_cols, agg_func)
    
    def analyze_trend(
        self, 
        time_series_df: pd.DataFrame, 
        time_col: str, 
        value_col: str
    ) -> dict:
        return self.trend_analyzer.run(time_series_df, time_col, value_col)
    
    def analyze_autocorrelation(
        self, 
        time_series_df: pd.DataFrame, 
        value_col: str,
        max_lags: int = 20
    ) -> dict:
        return self.autocorrelation_analyzer.run(time_series_df, value_col, max_lags)
    
    def detect_seasonality(
        self, 
        time_series_df: pd.DataFrame, 
        value_col: str,
        period: int = 52
    ) -> dict:
        return self.seasonality_detector.run(time_series_df, value_col, period)
    
    def plot_time_series(
        self, 
        time_series_df: pd.DataFrame, 
        time_col: str, 
        value_cols: list[str],
        trend_stats: Optional[dict] = None,
        figsize: tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        return self.time_series_plotter.run(
            time_series_df, time_col, value_cols, trend_stats, figsize
        )
    
    def plot_autocorrelation(
        self, 
        acf_dict: dict,
        title: str = "Autocorrelation (ACF)",
        figsize: tuple[int, int] = (10, 5)
    ) -> plt.Figure:
        return self.autocorrelation_plotter.run(acf_dict, title, figsize)