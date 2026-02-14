import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats


class DistributionPlotter:
    """
    Plots histograms with overlaid normal distribution and statistics.
    Public API: run().
    """
    
    def run(
        self,
        df: pd.DataFrame,
        columns: list[str],
        bins: int = 50,
        figsize: tuple[int, int] = (14, 5),
        stats_dict: dict = None
    ) -> plt.Figure:
        """
        Public API. Creates histogram plots for specified columns.
        
        Args:
            df: DataFrame with data
            columns: list of column names to plot
            bins: number of histogram bins
            figsize: figure size
            stats_dict: optional dict from DistributionAnalyzer.run()
            
        Returns:
            matplotlib Figure object
        """
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, columns):
            series = df[col].dropna()
            
            # Histogram
            n, bins_edges, patches = ax.hist(
                series, 
                bins=bins, 
                density=True, 
                alpha=0.7, 
                edgecolor='black',
                label='Data'
            )
            
            # Overlay normal distribution
            mu, sigma = series.mean(), series.std()
            x = np.linspace(series.min(), series.max(), 100)
            ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Theoretical normal')
            
            # Add vertical lines for mean and median
            ax.axvline(mu, color='blue', linestyle='--', linewidth=1.5, label=f'Mean = {mu:.3f}')
            ax.axvline(series.median(), color='green', linestyle='--', linewidth=1.5, label=f'Median = {series.median():.3f}')
            
            # Labels and title
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {col}')
            ax.legend(fontsize=8)
            
            # Add stats text if provided
            if stats_dict and col in stats_dict:
                s = stats_dict[col]
                textstr = f"Skew: {s['skewness']:.3f}\nKurt: {s['kurtosis']:.3f}\nN: {s['count']}"
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        return fig