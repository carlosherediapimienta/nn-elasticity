import pandas as pd
from scipy import stats


class TrendAnalyzer:
    """
    Analyzes temporal trends using linear regression.
    Single responsibility: detect if there's upward/downward trend.
    Public API: run().
    """
    
    def run(
        self,
        time_series_df: pd.DataFrame,
        time_col: str,
        value_col: str
    ) -> dict:
        """
        Public API. Fits linear trend and returns slope and statistics.
        
        Args:
            time_series_df: DataFrame with aggregated time series
            time_col: column name for time
            value_col: column name for values
            
        Returns:
            dict with: slope, intercept, r_squared, pvalue, trend_direction
        """
        
        x = time_series_df[time_col].values
        y = time_series_df[value_col].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if p_value < 0.05:
            if slope > 0:
                trend = 'upward'
            elif slope < 0:
                trend = 'downward'
            else:
                trend = 'flat'
        else:
            trend = 'no significant trend'
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'pvalue': float(p_value),
            'std_err': float(std_err),
            'trend_direction': trend
        }