import pandas as pd
from .multiproduct import (
    UPCSelector,
    StoreSelector,
    CompleteObservationFilter,
    MultiProductPivoter,
)


class MultiProductBuilder:
    """
    Orchestrates the transformation from long format to multi-product wide format.
    Lags and rolling features are precomputed in the source dataset.
    Public API: fit(train_df, n), transform(df)
    """

    def __init__(self, min_coverage: float = 0.5, min_products: int = 1):
        self.selector       = UPCSelector()
        self.store_selector = StoreSelector()
        self.filter         = CompleteObservationFilter(
            min_coverage=min_coverage,
            min_products=min_products,
        )
        self.pivoter = MultiProductPivoter()

    def fit(self, train_df: pd.DataFrame, n=None, upcs=None, stores=None) -> "MultiProductBuilder":
        self.selector.fit(train_df, n=n, upcs=upcs)
        self.store_selector.fit(train_df, n=n, stores=stores)

        train_filtered = self.filter.run(
            train_df,
            self.selector.selected_upcs,
            self.store_selector.selected_stores,
        )
        self.selector.selected_upcs = self.filter.valid_upcs
        self.n = len(self.selector.selected_upcs)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = self.filter.run(
            df,
            self.selector.selected_upcs,
            self.store_selector.selected_stores,
        )
        return self.pivoter.run(filtered, self.selector.selected_upcs)

    @property
    def selected_upcs(self) -> list:
        return self.selector.selected_upcs

    @property
    def selected_stores(self) -> list:
        return self.store_selector.selected_stores