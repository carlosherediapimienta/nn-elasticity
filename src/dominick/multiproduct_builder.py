import pandas as pd
from .multiproduct import PanelSelector, CompleteObservationFilter, MultiProductPivoter


class MultiProductBuilder:
    """
    fit_panel(df_sample): UPC/store selection on a frozen window (no val weeks).
    fit_impute(train_df): price medians / last observed on that train fold.
    transform(df): pivot + causal price fill.
    """

    def __init__(self, min_coverage: float = 0.5, min_products: int = 1):
        self.selector = PanelSelector()
        self.filter = CompleteObservationFilter(
            min_coverage=min_coverage, min_products=min_products,
        )
        self.pivoter = MultiProductPivoter()

    def fit_panel(self, df: pd.DataFrame, n_upcs: int | None = None,
                  upcs: list | None = None, stores: list | None = None):
        self.selector.fit(df, n_upcs=n_upcs, upcs=upcs, stores=stores)
        filtered = self.filter.run(
            df, self.selector.selected_upcs, self.selector.selected_stores,
        )
        self.selector.selected_upcs = self.filter.valid_upcs
        self.n = len(self.selector.selected_upcs)
        return self

    def fit_impute(self, train_df: pd.DataFrame):
        if not hasattr(self.selector, "selected_upcs"):
            raise RuntimeError("fit_panel() first")
        train_df = train_df[train_df["upc_code"].isin(self.selected_upcs)]
        train_df = train_df[train_df["store_code"].isin(self.selected_stores)]
        self.pivoter.fit(train_df, self.selected_upcs)
        return self

    def transform(self, df: pd.DataFrame, seed_from_train: bool = False) -> pd.DataFrame:
        df = df[df["upc_code"].isin(self.selected_upcs)]
        df = df[df["store_code"].isin(self.selected_stores)]
        return self.pivoter.transform(
            df, self.selected_upcs, seed_from_train=seed_from_train,
        )

    def make_fold_frames(self, train_long, val_long):
        self.fit_impute(train_long)
        return (
            self.transform(train_long, seed_from_train=False),
            self.transform(val_long, seed_from_train=True),
        )
        
    @property
    def selected_upcs(self) -> list:
        return self.selector.selected_upcs

    @property
    def selected_stores(self) -> list:
        return self.selector.selected_stores