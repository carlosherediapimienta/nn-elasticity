import pandas as pd
from .multiproduct import (
    PanelSelector,
    CompleteObservationFilter,
    MultiProductPivoter,
)


class MultiProductBuilder:
    """
    Orchestrates the transformation from long format to multi-product wide format.
    Public API: fit(df, n_upcs, upcs, stores), transform(df)
    """

    def __init__(self, min_coverage: float = 0.5, min_products: int = 1):
        self.selector = PanelSelector()
        self.filter   = CompleteObservationFilter(
            min_coverage=min_coverage,
            min_products=min_products,
        )
        self.pivoter = MultiProductPivoter()

    def fit(
        self,
        df: pd.DataFrame,
        n_upcs: int | None = None,
        upcs: list | None = None,
        stores: list | None = None,
    ) -> "MultiProductBuilder":
        self.selector.fit(df, n_upcs=n_upcs, upcs=upcs, stores=stores)

        filtered = self.filter.run(
            df,
            self.selector.selected_upcs,
            self.selector.selected_stores,
        )
        # El filtro puede reducir los UPCs (min_coverage); actualizamos
        self.selector.selected_upcs  = self.filter.valid_upcs
        self.n = len(self.selector.selected_upcs)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = self.filter.run(
            df,
            self.selector.selected_upcs,
            self.selector.selected_stores,
        )
        return self.pivoter.run(filtered, self.selector.selected_upcs)

    @property
    def selected_upcs(self) -> list:
        return self.selector.selected_upcs

    @property
    def selected_stores(self) -> list:
        return self.selector.selected_stores