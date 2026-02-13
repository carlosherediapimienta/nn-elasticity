import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

class CorrelationHistogramPlotter:
    """
    Dibuja histograma de correlaciones con mediana.
    API pública: run().
    """

    def run(
        self,
        corr_df: pd.DataFrame,
        corr_col: str = "corr",
        title: str = "Correlación de Spearman por tienda",
        xlabel: str = "Correlación de Spearman",
        ylabel: str = "Nº tiendas",
        bins: int = 25,
        figsize: tuple[int, int] = (8, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """API pública. Dibuja el histograma y retorna el Axes."""
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        series = corr_df[corr_col].dropna()
        ax.hist(series, bins=bins, edgecolor="black", alpha=0.7)
        med = series.median()
        ax.axvline(med, color="red", linestyle="--", linewidth=2, label=f"Mediana = {med:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.figure.tight_layout()
        return ax