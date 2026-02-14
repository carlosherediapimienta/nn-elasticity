import pandas as pd
import torch
from torch.utils.data import Dataset
from .schema import DatasetSchema
from ..preprocessing import ColumnConverter


class DemandDataset(Dataset):
    """
    Dataset for demand models.
    
    Converts DataFrame of pandas to dict of PyTorch tensors.
    
    Delegation:
    - DatasetSchema: define column structure
    - ColumnConverter: convert columns to tensors
    
    Public API: __len__(), __getitem__().
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema | None = None,
        converter: ColumnConverter | None = None,
        validate: bool = True
    ):
        """
        Args:
            df: DataFrame with data
            schema: column schema (optional, uses default if None)
            converter: column converter (optional)
            validate: if True, validate that all columns exist
        """
        self.schema = schema or DatasetSchema()
        self.converter = converter or ColumnConverter()
        
        # Validate DataFrame
        if validate:
            self.schema.validate_dataframe(df)
        
        # Convert all columns to tensors
        dtype_specs = self.schema.get_dtype_specs()
        self.tensors = self.converter.convert_columns(df, dtype_specs)
        
        # Validate that all tensors have the same size
        lengths = {k: len(v) for k, v in self.tensors.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"Tensors have inconsistent sizes: {lengths}")
        
        self._len = list(lengths.values())[0]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self._len
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return the sample at the given index.
        
        Args:
            idx: index of the sample
        
        Returns:
            dict with tensors of all features and target
        """
        return {
            key: tensor[idx]
            for key, tensor in self.tensors.items()
        }
    
    def get_column_tensor(self, column: str) -> torch.Tensor:
        """
        Return the tensor of a column.
        
        Args:
            column: name of the column
        
        Returns:
            tensor with all values of the column
        """
        if column not in self.tensors:
            raise KeyError(f"Column '{column}' not found in dataset")
        return self.tensors[column]