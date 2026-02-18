import pandas as pd

class LagFeatureBuilder:
    """
    Computes lag/rolling features within each (store_code, upc_code) group,
    sorted by week_id. All NaNs filled with 0.0.
    """

    @staticmethod
    def _weeks_since_promo(s: pd.Series) -> pd.Series:
        promo = (s == 1).astype(int)
        return s.groupby(promo.cumsum()).cumcount().astype(float)

    def fit(self, train_df: pd.DataFrame) -> "LagFeatureBuilder":
        self.fill_y = float(train_df["log_liters_sold"].mean())
        self.fill_x = float(train_df["log_price_per_liter"].mean())
        return self

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["on_promo"] = df["on_promo"].astype(float)
        g = df.sort_values(["store_code","upc_code","week_id"]).groupby(["store_code", "upc_code"])
        
        # A: Sales memory
        y = g["log_liters_sold"]
        df["lag_y_1"]           = y.shift(1)
        df["lag_y_2"]           = y.shift(2)
        df["lag_y_4"]           = y.shift(4)
        df["rolling_mean_y_4"]  = y.transform(lambda s: s.shift(1).rolling(4,  min_periods=1).mean())
        df["rolling_mean_y_13"] = y.transform(lambda s: s.shift(1).rolling(13, min_periods=1).mean())

        # Missingness flags (before filling)
        lag_y_cols = ["lag_y_1", "lag_y_2", "lag_y_4", "rolling_mean_y_4", "rolling_mean_y_13"]
        for c in lag_y_cols:
            df[c + "_missing"] = df[c].isna().astype(float)
        for c in lag_y_cols:
            df[c] = df[c].fillna(self.fill_y)
        
        # B: Price memory
        x = g["log_price_per_liter"]
        df["lag_x_1"]          = x.shift(1).fillna(self.fill_x)
        df["delta_x_1"]        = df["log_price_per_liter"] - x.shift(1).fillna(self.fill_x)
        df["rolling_mean_x_4"] = x.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()).fillna(self.fill_x)

        
        # C: Promotion memory
        p = g["on_promo"]
        df["lag_onpromo_1"]     = p.shift(1)
        df["lag_onpromo_2"]     = p.shift(2)
        df["weeks_since_promo"] = g["on_promo"].transform(
            lambda s: LagFeatureBuilder._weeks_since_promo(s.shift(1).fillna(0.0))
        )

        # D: Week gap (detects non-consecutive weeks)
        df["week_gap_1"] = g["week_id"].transform(
            lambda s: s.diff().fillna(0.0)
        )
        
        return df.fillna(0.0)