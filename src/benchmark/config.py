from dataclasses import dataclass, field
from typing import List

@dataclass
class ElasticityConfig:
    csv_path: str
    target_col: str = "log_liters_sold"
    price_col: str = "log_price_per_liter"
    control_cols: List[str] = field(default_factory=lambda: [
        "on_promo",
        "sin_52",
        "cos_52",
        "sin_26",
        "cos_26",
        "promo_intensity_store_week",
    ])