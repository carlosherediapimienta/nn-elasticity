import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from sklearn.linear_model import LinearRegression


class ScatterRegressionPlotter:
    """
    Creates scatter plot with OLS regression line and R².
    Public API: run().
    """
    
    def run(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        sample_size: Optional[int] = None,
        figsize: tuple[int, int] = (10, 7),
        stats_dict: dict = None,
        alpha: float = 0.3
    ) -> plt.Figure:
        """
        Public API. Creates scatter plot with regression line.
        
        Args:
            df: DataFrame with data
            x_col: name of x variable (predictor)
            y_col: name of y variable (response)
            sample_size: if provided, plot random sample (useful for large datasets)
            figsize: figure size
            stats_dict: optional dict from GlobalCorrelationAnalyzer.run()
            alpha: transparency of points
            
        Returns:
            matplotlib Figure object
        """
        # Clean data
        mask = df[x_col].notna() & df[y_col].notna()
        plot_df = df.loc[mask, [x_col, y_col]].copy()
        
        # Sample if needed
        if sample_size and len(plot_df) > sample_size:
            plot_df = plot_df.sample(n=sample_size, random_state=42)
        
        x = plot_df[x_col].values.reshape(-1, 1)
        y = plot_df[y_col].values
        
        # Fit regression
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        r_squared = model.score(x, y)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(x, y, alpha=alpha, s=10, label='Observations')
        
        # Regression line
        x_line = np.array([x.min(), x.max()]).reshape(-1, 1)
        y_line = model.predict(x_line)
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'OLS Regression (R² = {r_squared:.4f})')
        
        # Labels
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'Relationship between {x_col} and {y_col}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        if stats_dict:
            textstr = '\n'.join([
                f"Pearson r = {stats_dict['pearson']['correlation']:.4f}",
                f"Spearman ρ = {stats_dict['spearman']['correlation']:.4f}",
                f"R² = {stats_dict['regression']['r_squared']:.4f}",
                f"β (elasticity) = {stats_dict['regression']['slope']:.4f}",
                f"p-value < 0.001" if stats_dict['regression']['pvalue'] < 0.001 else f"p-value = {stats_dict['regression']['pvalue']:.4f}",
                f"n = {stats_dict['n_obs']:,}"
            ])
        else:
            textstr = f"R² = {r_squared:.4f}\nβ = {model.coef_[0]:.4f}\nn = {len(x):,}"
        
        # Position text box
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
        return fig