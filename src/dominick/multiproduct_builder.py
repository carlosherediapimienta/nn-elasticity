import pandas as pd
from .multiproduct import (
    UPCSelector,
    StoreSelector,
    CompleteObservationFilter,
    MultiProductPivoter,
    MultiProductLagBuilder,
)

class MultiProductBuilder:
    """
    Orchestrates the transformation from long format Dominick to multi-product wide format.
    Public API: fit(train_df, n), transform(df)
    """
    def __init__(self, min_coverage: float = 0.5, min_products: int = 1):
        self.selector       = UPCSelector()
        self.store_selector = StoreSelector()
        self.filter         = CompleteObservationFilter(
            min_coverage=min_coverage,
            min_products=min_products,   
        )
        self.pivoter     = MultiProductPivoter()
        self.lag_builder = None 

    def fit(self, train_df, n=None, upcs=None, stores=None):
        self.selector.fit(train_df, n=n, upcs=upcs)
        self.store_selector.fit(train_df, n=n, stores=stores)

        train_filtered = self.filter.run(
            train_df,
            self.selector.selected_upcs,
            self.store_selector.selected_stores
        )
        # Propagar las UPCs que sobrevivieron el filtro de cobertura
        self.selector.selected_upcs = self.filter.valid_upcs
        self.n = len(self.selector.selected_upcs)

        train_wide = self.pivoter.run(train_filtered, self.selector.selected_upcs)
        self.lag_builder = MultiProductLagBuilder().fit(train_wide, self.n)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = self.filter.run(df, self.selector.selected_upcs, self.store_selector.selected_stores)
        wide     = self.pivoter.run(filtered, self.selector.selected_upcs)
        return self.lag_builder.build(wide)

    @property
    def selected_upcs(self) -> list:
        return self.selector.selected_upcs

    @property
    def selected_stores(self) -> list:
        return self.store_selector.selected_stores