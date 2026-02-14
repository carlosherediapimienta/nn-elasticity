import pandas as pd
import matplotlib.pyplot as plt


class BoxPlotter:
    """
    Creates box plots for outlier visualization.
    Public API: run().
    """
    
    def run(
        self,
        df: pd.DataFrame,
        columns: list[str],
        figsize: tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """
        Public API. Creates box plots for specified columns.
        
        Args:
            df: DataFrame with data
            columns: list of column names to plot
            figsize: figure size
            
        Returns:
            matplotlib Figure object
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for ax, col in zip(axes, columns):
            series = df[col].dropna()
            
            bp = ax.boxplot(series, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Add mean marker
            mean_val = series.mean()
            ax.plot([1], [mean_val], marker='o', color='red', markersize=8, label=f'Mean = {mean_val:.3f}')
            
            ax.set_ylabel(col)
            ax.set_title(f'Box plot of {col}')
            ax.set_xticks([1])
            ax.set_xticklabels([col])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        fig.tight_layout()
        return fig