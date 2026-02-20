import pandas as pd
from .multiproduct import (
    UPCSelector,
    CompleteObservationFilter,
    MultiProductPivoter,
    MultiProductLagBuilder,
)

class MultiProductBuilder:
    """
    Orchestrates the transformation from long format Dominick to multi-product wide format.
    Public API: fit(train_df, n), transform(df)
    """
    def __init__(self):
        self.selector = UPCSelector()
        self.filter   = CompleteObservationFilter()
        self.pivoter  = MultiProductPivoter()
        self.lag_builder = None   

    def fit(self, train_df: pd.DataFrame, n: int | None = None, upcs: list | None = None) -> "MultiProductBuilder":
        self.selector.fit(train_df, n=n, upcs=upcs)         
        self.n = len(self.selector.selected_upcs)             
        train_filtered = self.filter.run(train_df, self.selector.selected_upcs)
        train_wide     = self.pivoter.run(train_filtered, self.selector.selected_upcs)
        self.lag_builder = MultiProductLagBuilder().fit(train_wide, self.n)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = self.filter.run(df, self.selector.selected_upcs)
        wide     = self.pivoter.run(filtered, self.selector.selected_upcs)
        return self.lag_builder.build(wide)

    @property
    def selected_upcs(self) -> list:
        return self.selector.selected_upcs