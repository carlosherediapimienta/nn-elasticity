import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


class GlobalCorrelationAnalyzer:
    """
    Calculates global correlations (Pearson, Spearman) and linear regression R².
    Public API: run().
    """
    
    def run(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str
    ) -> dict:
        """
        Public API. Returns dict with global correlation metrics.
        
        Args:
            df: DataFrame with data
            x_col: name of x variable (predictor)
            y_col: name of y variable (response)
            
        Returns:
            dict with structure:
            {
                'pearson': {'correlation': float, 'pvalue': float},
                'spearman': {'correlation': float, 'pvalue': float},
                'regression': {
                    'r_squared': float,
                    'slope': float,
                    'intercept': float,
                    'std_err': float,
                    'pvalue': float
                },
                'n_obs': int
            }
        """
        
        # Clean data
        mask = df[x_col].notna() & df[y_col].notna()
        x = df.loc[mask, x_col].values.reshape(-1, 1)
        y = df.loc[mask, y_col].values
        n_obs = len(x)
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(x.flatten(), y)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(x.flatten(), y)
        
        # Linear regression
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        
        # R² and other metrics
        r_squared = model.score(x, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Standard error and p-value for slope
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n_obs - 2)
        x_mean = x.mean()
        se_slope = np.sqrt(mse / np.sum((x - x_mean)**2))
        t_stat = slope / se_slope
        slope_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 2))
        
        return {
            'pearson': {
                'correlation': float(pearson_r),
                'pvalue': float(pearson_p)
            },
            'spearman': {
                'correlation': float(spearman_r),
                'pvalue': float(spearman_p)
            },
            'regression': {
                'r_squared': float(r_squared),
                'slope': float(slope),
                'intercept': float(intercept),
                'std_err': float(se_slope),
                'pvalue': float(slope_pvalue)
            },
            'n_obs': int(n_obs)
        }