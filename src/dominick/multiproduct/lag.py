import pandas as pd

class MultiProductLagBuilder:
    """
    Calculates lags for the wide format (n products per row).
    Prices are forward-filled before lag computation to avoid NaN propagation.
    Missing demand lags are filled with the training mean.
    Group: (store_code), sorted by week_id.
    Public API: fit(train_df, n) → self, build(df) → df with lags
    """

    def fit(self, train_df: pd.DataFrame, n: int) -> "MultiProductLagBuilder":
        self.n = n
        self.fill_y = []
        self.fill_x = []
        for i in range(n):
            mask = train_df[f"obs_mask_{i}"] == 1 if f"obs_mask_{i}" in train_df.columns else slice(None)
            self.fill_y.append(float(train_df.loc[mask, f"log_liters_{i}"].mean()))
            self.fill_x.append(float(train_df[f"log_price_{i}"].mean()))
        return self

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(["store_code", "week_id"])
        g = df.groupby("store_code")

        new_cols = {}

        for i in range(self.n):
            y = g[f"log_liters_{i}"]
            new_cols[f"lag_y_{i}_1"] = y.shift(1).fillna(self.fill_y[i])
            new_cols[f"rolling_mean_y_{i}_4"] = y.transform(
                lambda s: s.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(self.fill_y[i])

            x = g[f"log_price_{i}"]
            lag_x = x.shift(1).fillna(self.fill_x[i])
            new_cols[f"lag_x_{i}_1"]   = lag_x
            new_cols[f"delta_x_{i}_1"] = df[f"log_price_{i}"] - lag_x

        new_cols["week_gap_1"] = g["week_id"].transform(
            lambda s: s.diff().fillna(0.0)
        )

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df.fillna(0.0)