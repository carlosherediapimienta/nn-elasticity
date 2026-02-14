import pandas as pd


class TimeSeriesAggregator:
    """
    Aggregates data by time periods (e.g., week_id).
    Single responsibility: group by time and calculate aggregates.
    Public API: run().
    """
    
    def run(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_cols: list[str],
        agg_func: str = 'mean'
    ) -> pd.DataFrame:
        """
        Public API. Aggregates values by time periods.
        
        Args:
            df: DataFrame with data
            time_col: column name for time periods (e.g., 'week_id')
            value_cols: list of columns to aggregate
            agg_func: aggregation function ('mean', 'sum', 'median', etc.)
            
        Returns:
            DataFrame with time_col as index and aggregated values
        """
        agg_dict = {col: agg_func for col in value_cols}
        result = df.groupby(time_col)[value_cols].agg(agg_func).reset_index()
        result = result.sort_values(time_col)
        return result