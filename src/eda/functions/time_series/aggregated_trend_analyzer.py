import pandas as pd
import numpy as np


class AggregatedTrendAnalyzer:
    """
    Agrega el panel por semana y calcula la correlación de week_rank con
    precio medio, demanda agregada y promo_rate para evaluar riesgo de
    elasticidad espuria por tendencias temporales.
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        time_col: str = "week_id",
        price_col: str = "log_price_per_liter",
        demand_raw_col: str = "liters_sold",
        promo_col: str = "on_promo",
    ) -> dict:
        """
        Args:
            df: DataFrame con grano (store, upc, week) y columnas de precio, demanda y promo.
            time_col: columna de semana.
            price_col: columna de precio en log (se agrega como media por semana).
            demand_raw_col: columna de cantidad en nivel (litros); se suma por semana y se toma log.
            promo_col: columna 0/1 de promo (se agrega como media = promo_rate).

        Returns:
            dict con:
                - correlations: dict con
                    * corr_week_rank_mean_log_price
                    * corr_week_rank_log_total_demand
                    * corr_week_rank_promo_rate
                - aggregated_series: DataFrame con columnas
                    * time_col
                    * week_rank (1, 2, ..., T)
                    * mean_log_price
                    * log_total_demand
                    * promo_rate
        """
        for col in [time_col, price_col, demand_raw_col, promo_col]:
            if col not in df.columns:
                raise ValueError(f"Columna '{col}' no encontrada en el DataFrame.")

        agg = df.groupby(time_col).agg(
            mean_log_price=(price_col, "mean"),
            total_demand=(demand_raw_col, "sum"),
            promo_rate=(promo_col, "mean"),
        ).reset_index()

        agg = agg.sort_values(time_col).reset_index(drop=True)
        agg["week_rank"] = np.arange(1, len(agg) + 1, dtype=float)

        total_demand = agg["total_demand"].values
        safe_demand = np.where(total_demand > 0, total_demand, np.nan)
        agg["log_total_demand"] = np.log(np.clip(safe_demand, 1e-12, None))

        # Calcular correlaciones solo sobre semanas con datos reales (sin NaN)
        agg_clean = agg.dropna(subset=["mean_log_price", "log_total_demand", "promo_rate"])

        rank = agg_clean["week_rank"]
        r1 = rank.corr(agg_clean["mean_log_price"])
        r2 = rank.corr(agg_clean["log_total_demand"])
        r3 = rank.corr(agg_clean["promo_rate"])

        correlations = {
            "corr_week_rank_mean_log_price": float(r1),
            "corr_week_rank_log_total_demand": float(r2),
            "corr_week_rank_promo_rate": float(r3),
        }

        out_series = agg[[time_col, "week_rank", "mean_log_price", "log_total_demand", "promo_rate"]].copy()
        out_series = out_series.rename(columns={"log_total_demand": "log_total_liters_sold"})

        return {
            "correlations": correlations,
            "aggregated_series": out_series,
            "time_col": time_col,
        }