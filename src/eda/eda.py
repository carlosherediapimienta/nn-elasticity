import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from .functions.analyzer import NanAnalyzer, CorrelationAnalyzer
from .functions.plotter import CorrelationHistogramPlotter


class EDA:
    """
    Orchestrator of EDA analysis. Coordinates NanAnalyzer, CorrelationAnalyzer
    and CorrelationHistogramPlotter using only their public API (run).
    Public API of this class: run().
    """

    def __init__(self):
        self.nan_analyzer = NanAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.histogram_plotter = CorrelationHistogramPlotter()

    def analyze_nans(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        return self.nan_analyzer.run(df, columns=columns)
    
    def analyze_correlations(self, df: pd.DataFrame, x_col: str, y_col: str, group_col: str = "store_code", min_obs: int = 30) -> pd.DataFrame:
        return self.correlation_analyzer.run(df, x_col, y_col, group_col=group_col, min_obs=min_obs)
    
    def plot_correlations(self, corr_df: pd.DataFrame, title: str = "Correlación de Spearman por tienda", xlabel: str = "Correlación de Spearman", ylabel: str = "Nº tiendas", bins: int = 25, figsize: tuple[int, int] = (8, 5), ax: Optional[plt.Axes] = None) -> plt.Axes:
        return self.histogram_plotter.run(corr_df, title=title, xlabel=xlabel, ylabel=ylabel, bins=bins, figsize=figsize, ax=ax)