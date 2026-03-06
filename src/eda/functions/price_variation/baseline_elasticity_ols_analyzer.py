import pandas as pd
import numpy as np
from scipy import stats


class BaselineElasticityOLSAnalyzer:
    """
    Baseline OLS log-log: regresión de log(demanda) sobre log(precio).
    - Naive (sin FE): elasticidad = coeficiente de log(precio).
    - 2-way FE (store×UPC + week): double demeaning, luego OLS.
    NO causal; útil como sanity check de magnitud.
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        price_col: str = "log_price_per_liter",
        demand_col: str = "log_liters_sold",
        store_col: str = "store_code",
        upc_col: str = "upc_code",
        week_col: str = "week_id",
    ) -> dict:
        """
        Args:
            df: DataFrame con grano (store, upc, week) y columnas precio y demanda en log.
            price_col: columna de precio en log.
            demand_col: columna de demanda en log.
            store_col, upc_col: para definir entidad (store×UPC).
            week_col: columna de tiempo para FE temporales.

        Returns:
            dict con:
                - ols_naive: elasticity (slope), r_squared, n_obs, se, pvalue.
                - ols_2way_fe: elasticity (slope), r_squared, n_obs, se, pvalue.
        """
        for col in [price_col, demand_col, store_col, upc_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        df_work = df[[store_col, upc_col, week_col, price_col, demand_col]].dropna(
            subset=[price_col, demand_col]
        )
        n_obs = len(df_work)

        x_naive = df_work[price_col].values.reshape(-1, 1)
        y_naive = df_work[demand_col].values

        # --- OLS naive ---
        slope_naive, intercept_naive, r2_naive, se_naive, pval_naive = self._ols_slope_stats(
            x_naive, y_naive
        )

        # --- 2-way FE: double demean por (store, upc) y por week ---
        entity = df_work[[store_col, upc_col]].astype(str).agg("_".join, axis=1)
        time_vals = df_work[week_col].values

        y = df_work[demand_col].values
        x = df_work[price_col].values

        y_dm = self._double_demean(y, entity, time_vals)
        x_dm = self._double_demean(x, entity, time_vals)

        mask = np.isfinite(y_dm) & np.isfinite(x_dm)
        x_fe = x_dm[mask].reshape(-1, 1)
        y_fe = y_dm[mask]
        n_fe = len(y_fe)

        slope_fe, _, r2_fe, se_fe, pval_fe = self._ols_slope_stats(x_fe, y_fe)

        return {
            "ols_naive": {
                "elasticity": float(slope_naive),
                "r_squared": float(r2_naive),
                "n_obs": n_obs,
                "se": float(se_naive),
                "pvalue": float(pval_naive),
            },
            "ols_2way_fe": {
                "elasticity": float(slope_fe),
                "r_squared": float(r2_fe),
                "n_obs": n_fe,
                "se": float(se_fe),
                "pvalue": float(pval_fe),
            },
            "price_col": price_col,
            "demand_col": demand_col,
        }

    def _double_demean(
        self, values: np.ndarray, entity: pd.Series, time_vals: np.ndarray
    ) -> np.ndarray:
        """Double demean: values - entity_mean - time_mean + overall_mean."""
        df = pd.DataFrame({"y": values, "entity": entity.values, "time": time_vals})
        overall = df["y"].mean()
        entity_means = df.groupby("entity")["y"].transform("mean")
        time_means = df.groupby("time")["y"].transform("mean")
        return (df["y"] - entity_means - time_means + overall).values

    def _ols_slope_stats(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        """OLS slope, intercept, R², SE(slope), p-value(slope)."""
        n = len(y)
        if n < 3:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        x_flat = x.flatten()
        slope = np.cov(x_flat, y)[0, 1] / np.var(x_flat)
        intercept = np.mean(y) - slope * np.mean(x_flat)
        y_hat = intercept + slope * x_flat
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mse = ss_res / (n - 2) if n > 2 else 0.0
        se_slope = np.sqrt(mse / np.sum((x_flat - np.mean(x_flat)) ** 2)) if np.var(x_flat) > 0 else np.nan
        t = slope / se_slope if se_slope and se_slope > 0 else 0.0
        pval = 2 * (1 - stats.t.cdf(abs(t), n - 2))
        return float(slope), float(intercept), float(r2), float(se_slope), float(pval)