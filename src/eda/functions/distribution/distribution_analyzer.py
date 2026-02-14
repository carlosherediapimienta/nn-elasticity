import pandas as pd
from scipy import stats


class DistributionAnalyzer:
    """
    Analyzes distributions of numeric variables: histograms, descriptive stats, normality tests.
    Public API: run().
    """
    
    def run(self, df: pd.DataFrame, columns: list[str]) -> dict:
        """
        Public API. Returns dict with statistical analysis for each column.
        
        Args:
            df: DataFrame with data
            columns: list of column names to analyze
            
        Returns:
            dict with structure:
            {
                'column_name': {
                    'mean': float,
                    'median': float,
                    'std': float,
                    'min': float,
                    'max': float,
                    'q25': float,
                    'q75': float,
                    'skewness': float,
                    'kurtosis': float,
                    'normality_test': {
                        'statistic': float,
                        'pvalue': float,
                        'test_name': str
                    }
                }
            }
        """
    
        results = {}
        for col in columns:
            series = df[col].dropna()
            
            # Descriptive stats
            desc = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'count': len(series)
            }
            
            # Normality test (Kolmogorov-Smirnov for large samples, Shapiro for small)
            if len(series) > 5000:
                stat, pval = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                test_name = 'Kolmogorov-Smirnov'
            else:
                stat, pval = stats.shapiro(series)
                test_name = 'Shapiro-Wilk'
            
            desc['normality_test'] = {
                'statistic': float(stat),
                'pvalue': float(pval),
                'test_name': test_name,
                'is_normal': pval > 0.05  # alpha = 0.05
            }
            
            results[col] = desc
        
        return results