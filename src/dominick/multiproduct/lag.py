import pandas as pd

class MultiProductLagBuilder:
    """
    Calculates lags for the wide format (n products per row).
    Group: (store_code), sorted by week_id.
    Public API: fit(train_df) → self, build(df) → df with lags
    """
    def fit(self, train_df: pd.DataFrame, n: int) -> "MultiProductLagBuilder":
        self.n = n
        self.fill_y = [
            float(train_df[f"log_liters_{i}"].mean()) for i in range(n)
        ]
        self.fill_x = [
            float(train_df[f"log_price_{i}"].mean()) for i in range(n)
        ]
        return self

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        g = df.sort_values(["store_code", "week_id"]).groupby("store_code")

        for i in range(self.n):
            y = g[f"log_liters_{i}"]
            df[f"lag_y_{i}_1"] = y.shift(1).fillna(self.fill_y[i])
            df[f"rolling_mean_y_{i}_4"] = y.transform(
                lambda s: s.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(self.fill_y[i])

            x = g[f"log_price_{i}"]
            df[f"lag_x_{i}_1"]   = x.shift(1).fillna(self.fill_x[i])
            df[f"delta_x_{i}_1"] = df[f"log_price_{i}"] - x.shift(1).fillna(self.fill_x[i])

        df["week_gap_1"] = g["week_id"].transform(lambda s: s.diff().fillna(0.0))
        return df.fillna(0.0)