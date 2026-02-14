import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class TimeSeriesPlotter:
    """
    Plots time series with trend lines.
    Single responsibility: visualize temporal evolution.
    Public API: run().
    """
    
    def run(
        self,
        time_series_df: pd.DataFrame,
        time_col: str,
        value_cols: list[str],
        trend_stats: Optional[dict] = None,
        figsize: tuple[int, int] = (14, 6)
    ) -> plt.Figure:
        """
        Public API. Creates time series plot with optional trend line.
        
        Args:
            time_series_df: DataFrame with time series
            time_col: column name for time
            value_cols: list of columns to plot
            trend_stats: optional dict from TrendAnalyzer.run() (one per value_col)
            figsize: figure size
            
        Returns:
            matplotlib Figure object
        """
        n_cols = len(value_cols)
        fig, axes = plt.subplots(n_cols, 1, figsize=(figsize[0], figsize[1] * n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, value_cols):
            x = time_series_df[time_col].values
            y = time_series_df[col].values
            
            # Plot time series
            ax.plot(x, y, 'b-', linewidth=1.5, marker='o', markersize=3, label='Data')
            
            # Add trend line if provided
            if trend_stats and col in trend_stats:
                ts = trend_stats[col]
                trend_line = ts['intercept'] + ts['slope'] * x
                ax.plot(x, trend_line, 'r--', linewidth=2, 
                       label=f"Trend ({ts['trend_direction']})")
                
                # Add statistics text
                textstr = '\n'.join([
                    f"Slope: {ts['slope']:.6f}",
                    f"R²: {ts['r_squared']:.4f}",
                    f"p-value: {ts['pvalue']:.4e}",
                    f"Trend: {ts['trend_direction']}"
                ])
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel(time_col, fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.set_title(f'Time Series: {col}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig