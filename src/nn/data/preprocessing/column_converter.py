import numpy as np
import pandas as pd
import torch


class ColumnConverter:
    """
    Convert columns of DataFrame to PyTorch tensors.
    Public API: to_tensor().
    """
    
    def to_tensor(
        self,
        values: np.ndarray | pd.Series,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Convert values to PyTorch tensor.
        
        Args:
            values: array or Series to convert
            dtype: type of the resulting tensor
        
        Returns:
            PyTorch tensor
        """
        if isinstance(values, pd.Series):
            values = values.values
        return torch.tensor(values, dtype=dtype)
    
    def convert_columns(
        self,
        df: pd.DataFrame,
        column_specs: dict[str, torch.dtype]
    ) -> dict[str, torch.Tensor]:
        """
        Convert multiple columns according to specifications.
        
        Args:
            df: DataFrame with data
            column_specs: dict with {column_name: dtype}
        
        Returns:
            dict with {column_name: tensor}
        """
        result = {}
        for col, dtype in column_specs.items():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            result[col] = self.to_tensor(df[col], dtype)
        return result