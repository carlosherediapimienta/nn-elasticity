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