import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf


class SeasonalityDetector:
    """
    Detects seasonal patterns in time series.
    Single responsibility: identify recurring patterns.
    Public API: run().
    """
    
    def run(
        self,
        time_series_df: pd.DataFrame,
        value_col: str,
        period: int = 52  # 52 weeks = 1 year
    ) -> dict:
        """
        Public API. Tests for seasonality using autocorrelation at seasonal lag.
        
        Args:
            time_series_df: DataFrame with time series
            value_col: column name for values
            period: expected seasonal period (e.g., 52 for yearly with weekly data)
            
        Returns:
            dict with: seasonal_acf, is_seasonal, period_tested
        """
        
        values = time_series_df[value_col].values
        n = len(values)
        
        if n < period + 1:
            return {
                'seasonal_acf': None,
                'is_seasonal': False,
                'period_tested': period,
                'note': f'Not enough data (n={n}) to test period {period}'
            }
        
        # Compute ACF up to seasonal lag
        acf_values = acf(values, nlags=period, fft=False)
        seasonal_acf = float(acf_values[period])
        
        # Confidence bound
        confidence_bound = 1.96 / np.sqrt(n)
        
        return {
            'seasonal_acf': seasonal_acf,
            'confidence_bound': float(confidence_bound),
            'is_seasonal': abs(seasonal_acf) > confidence_bound,
            'period_tested': period
        }