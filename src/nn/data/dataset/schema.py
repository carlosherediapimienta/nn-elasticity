import torch
from dataclasses import dataclass, field


@dataclass
class DatasetSchema:
    """
    Define the schema of the dataset columns.
    Facilitates configuration and validation.
    """
    
    # Categorical columns (entity IDs)
    categorical_cols: list[str] = field(default_factory=lambda: [
        "store_code",
        "upc_code",
        "week_id"
    ])
    
    # Promotion columns
    promo_cols: list[str] = field(default_factory=lambda: [
        "on_promo",
        "promo_B",
        "promo_C",
        "promo_S"
    ])
    
    # Numerical features
    numeric_cols: list[str] = field(default_factory=lambda: [
        "liters_per_upc",
        "log_price_per_liter"
    ])
    
    # Target column
    target_col: str = "log_liters_sold"
    
    def get_all_required_columns(self) -> list[str]:
        """Return all required columns."""
        return (
            self.categorical_cols +
            self.promo_cols +
            self.numeric_cols +
            [self.target_col]
        )
    
    def validate_dataframe(self, df) -> None:
        """
        Validate that the DataFrame has all required columns.
        
        Args:
            df: DataFrame to validate
        
        Raises:
            ValueError: if columns are missing
        """
        required = set(self.get_all_required_columns())
        available = set(df.columns)
        missing = required - available
        
        if missing:
            raise ValueError(
                f"Columns missing in DataFrame: {sorted(missing)}"
            )
    
    def get_dtype_specs(self) -> dict[str, torch.dtype]:
        """
        Return data type specifications.
        
        Returns:
            dict with {column_name: torch.dtype}
        """
        specs = {}
        
        # Categorical → long
        for col in self.categorical_cols:
            specs[col] = torch.long
        
        # Promotion → float32
        for col in self.promo_cols:
            specs[col] = torch.float32
        
        # Numerical → float32
        for col in self.numeric_cols:
            specs[col] = torch.float32
        
        # Target column → float32
        specs[self.target_col] = torch.float32
        
        return specs