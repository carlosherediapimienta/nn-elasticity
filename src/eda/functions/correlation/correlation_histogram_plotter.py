import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class CorrelationHistogramPlotter:
    """
    Draws histogram of correlations with median.
    Public API: run().
    """

    def run(
        self,
        corr_df: pd.DataFrame,
        corr_col: str = "corr",
        title: str = "Spearman correlation by store",
        xlabel: str = "Spearman correlation",
        ylabel: str = "Number of stores",
        bins: int = 25,
        figsize: tuple[int, int] = (8, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Public API. Draws the histogram and returns the Axes."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        series = corr_df[corr_col].dropna()
        ax.hist(series, bins=bins, edgecolor="black", alpha=0.7)
        med = series.median()
        ax.axvline(med, color="red", linestyle="--", linewidth=2, label=f"Median = {med:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.figure.tight_layout()
        return ax