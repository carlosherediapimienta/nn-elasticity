import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiProductDataset(Dataset):
    """
    Dataset for n-product wide format (output of MultiProductBuilder).
    Each sample: (store, week) with n prices, n demands and per-product lags.
    Shared by nn and nn_baseline.
    Public API: __len__(), __getitem__()
    """

    PROMO_COLS = ["on_promo", "promo_B", "promo_C", "promo_S"]

    def __init__(self, df: pd.DataFrame, n: int):
        """
        Args:
            df: wide-format DataFrame from MultiProductBuilder.transform()
            n:  number of products
        """
        self.n = n
        t = {}

        t["store_code"] = torch.tensor(df["store_code"].values, dtype=torch.long)
        t["week_id"]    = torch.tensor(df["week_id"].values,    dtype=torch.long)

        for col in self.PROMO_COLS:
            t[col] = torch.tensor(df[col].values, dtype=torch.float32)

        for i in range(n):
            t[f"log_price_{i}"]         = torch.tensor(df[f"log_price_{i}"].values,         dtype=torch.float32)
            t[f"log_liters_{i}"]        = torch.tensor(df[f"log_liters_{i}"].values,        dtype=torch.float32)
            t[f"lag_y_{i}_1"]           = torch.tensor(df[f"lag_y_{i}_1"].values,           dtype=torch.float32)
            t[f"rolling_mean_y_{i}_4"]  = torch.tensor(df[f"rolling_mean_y_{i}_4"].values,  dtype=torch.float32)
            t[f"lag_x_{i}_1"]           = torch.tensor(df[f"lag_x_{i}_1"].values,           dtype=torch.float32)
            t[f"delta_x_{i}_1"]         = torch.tensor(df[f"delta_x_{i}_1"].values,         dtype=torch.float32)

        t["week_gap_1"] = torch.tensor(df["week_gap_1"].values, dtype=torch.float32)

        self.tensors = t
        self._len    = len(df)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}