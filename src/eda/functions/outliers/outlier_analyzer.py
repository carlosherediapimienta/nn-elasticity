import pandas as pd


class OutlierAnalyzer:
    """
    Detects outliers using ±3 std method or IQR method.
    Public API: run().
    """
    
    def run(self, df: pd.DataFrame, columns: list[str], method: str = 'std', n_std: float = 3.0) -> pd.DataFrame:
        """
        Public API. Returns DataFrame with outlier statistics.
        
        Args:
            df: DataFrame with data
            columns: list of column names to analyze
            method: 'std' (±n std) or 'iqr' (interquartile range)
            n_std: number of std deviations for outlier threshold (default 3)
            
        Returns:
            DataFrame with columns: variable, n_outliers, pct_outliers, method
        """
        results = []
        
        for col in columns:
            series = df[col].dropna()
            
            if method == 'std':
                mean = series.mean()
                std = series.std()
                lower = mean - n_std * std
                upper = mean + n_std * std
                outliers = (series < lower) | (series > upper)
                method_desc = f'±{n_std} std'
            
            elif method == 'iqr':
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = (series < lower) | (series > upper)
                method_desc = 'IQR (1.5)'
            
            n_outliers = outliers.sum()
            pct_outliers = (n_outliers / len(series) * 100)
            
            results.append({
                'variable': col,
                'n_outliers': int(n_outliers),
                'pct_outliers': round(pct_outliers, 2),
                'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4),
                'method': method_desc
            })
        
        return pd.DataFrame(results)