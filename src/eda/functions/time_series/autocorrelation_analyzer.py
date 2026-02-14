import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf


class AutocorrelationAnalyzer:
    """
    Calculates autocorrelation (ACF) for time series.
    Single responsibility: measure temporal dependence.
    Public API: run().
    """
    
    def run(
        self,
        time_series_df: pd.DataFrame,
        value_col: str,
        max_lags: int = 20
    ) -> dict:
        """
        Public API. Computes autocorrelation function (ACF).
        
        Args:
            time_series_df: DataFrame with time series (must be sorted by time)
            value_col: column name for values
            max_lags: maximum number of lags to compute
            
        Returns:
            dict with: acf_values (array), lags (array), significant_lags (list)
        """
        
        values = time_series_df[value_col].values
        n = len(values)
        
        # Compute ACF
        acf_values = acf(values, nlags=min(max_lags, n-1), fft=False)
        lags = np.arange(len(acf_values))
        
        # Confidence interval (95%): ±1.96/sqrt(n)
        confidence_bound = 1.96 / np.sqrt(n)
        
        # Find significant lags (excluding lag 0 which is always 1)
        significant_lags = [
            int(lag) for lag in lags[1:] 
            if abs(acf_values[lag]) > confidence_bound
        ]
        
        return {
            'acf_values': acf_values,
            'lags': lags,
            'confidence_bound': float(confidence_bound),
            'significant_lags': significant_lags,
            'has_autocorrelation': len(significant_lags) > 0
        }