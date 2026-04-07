import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf


class AutocorrelationAnalyzer:
    """
    Calculates autocorrelation (ACF) for time series.
    Measures temporal dependence between values at different lags.
    Public API: run().
    """
    def run(self, time_series_df, value_col, max_lags=20):
        series = time_series_df[value_col].dropna()

        if len(series) == 0:
            raise ValueError(f"The column '{value_col}' has no non-NaN values.")

        n_original = len(time_series_df[value_col])
        n_clean = len(series)
        n_dropped = n_original - n_clean

        values = series.values
        n = len(values)

        acf_values = acf(values, nlags=min(max_lags, n - 1), fft=False)
        lags = np.arange(len(acf_values))

        confidence_bound = 1.96 / np.sqrt(n)
        significant_lags = [
            int(lag) for lag in lags[1:]
            if abs(acf_values[lag]) > confidence_bound
        ]

        return {
            "acf_values": acf_values,
            "lags": lags,
            "confidence_bound": float(confidence_bound),
            "significant_lags": significant_lags,
            "has_autocorrelation": len(significant_lags) > 0,
            "n_obs_used": n_clean,
            "n_obs_dropped_nan": n_dropped,
        }