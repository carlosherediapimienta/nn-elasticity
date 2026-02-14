import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def factorize_col(df, col):
    codes, uniques = pd.factorize(df[col], sort=True)
    return codes.astype(np.int64), uniques

class DemandDataset(Dataset):
    def __init__(self, df):
        self.store = torch.tensor(df["store_code"].values, dtype=torch.long)
        self.upc   = torch.tensor(df["upc_code"].values, dtype=torch.long)
        self.week  = torch.tensor(df["week_id"].values, dtype=torch.long)

        self.on_promo = torch.tensor(df["on_promo"].values, dtype=torch.float32)
        self.pB = torch.tensor(df["promo_B"].values, dtype=torch.float32)
        self.pC = torch.tensor(df["promo_C"].values, dtype=torch.float32)
        self.pS = torch.tensor(df["promo_S"].values, dtype=torch.float32)

        self.liters_per_upc = torch.tensor(df["liters_per_upc"].values, dtype=torch.float32)
        self.x = torch.tensor(df["log_price_per_liter"].values, dtype=torch.float32)
        self.y = torch.tensor(df["log_liters_sold"].values, dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "store_code": self.store[idx],
            "upc_code": self.upc[idx],
            "week_id": self.week[idx],
            "on_promo": self.on_promo[idx],
            "promo_B": self.pB[idx],
            "promo_C": self.pC[idx],
            "promo_S": self.pS[idx],
            "liters_per_upc": self.liters_per_upc[idx],
            "log_price_per_liter": self.x[idx],
            "log_liters_sold": self.y[idx],
        }


def build_price_spline_from_train(x_train_np, K=16):
    x_mean = float(x_train_np.mean())
    x_std  = float(x_train_np.std() + 1e-6)
    qs = np.linspace(0.05, 0.95, K)
    knots = torch.tensor(np.quantile(x_train_np, qs), dtype=torch.float32)
    return knots, x_mean, x_std
