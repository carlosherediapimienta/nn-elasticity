import numpy as np
import pandas as pd


class ColumnEncoder:
    """
    Encode categorical columns into numerical codes.
    Public API: factorize().
    """
    
    def factorize(
        self,
        df: pd.DataFrame,
        column: str,
        sort: bool = True
    ) -> tuple[np.ndarray, pd.Index]:
        """
        Factorize a categorical column.
        
        Args:
            df: DataFrame with data
            column: name of the column to factorize
            sort: if True, sort the categories before encoding
        
        Returns:
            codes: array of numerical codes (int64)
            categories: index with unique categories
        """
        codes, categories = pd.factorize(df[column], sort=sort)
        return codes.astype(np.int64), categories
    
    def factorize_multiple(
        self,
        df: pd.DataFrame,
        columns: list[str],
        sort: bool = True
    ) -> dict[str, tuple[np.ndarray, pd.Index]]:
        """
        Factorize multiple columns.
        
        Args:
            df: DataFrame with data
            columns: list of columns to factorize
            sort: if True, sort the categories before encoding
        
        Returns:
            dict with {column_name: (codes, categories)}
        """
        result = {}
        for col in columns:
            result[col] = self.factorize(df, col, sort=sort)
        return result