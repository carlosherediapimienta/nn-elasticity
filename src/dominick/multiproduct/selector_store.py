import pandas as pd

class StoreSelector:
    """
    Selects stores for the multi-product pipeline.
    - If explicit stores are passed: use those (in that order).
    - If stores=None: use the n with most observations in train.
    - If stores=None and n=None: use ALL stores from the dataset.
    """

    def fit(
        self,
        train_df: pd.DataFrame,
        n: int | None = None,
        stores: list | None = None,
    ) -> "StoreSelector":

        if stores is not None:
            self.selected_stores = list(stores)

        elif n is None:
            self.selected_stores = train_df["store_code"].unique().tolist()

        else:
            self.selected_stores = (
                train_df.groupby("store_code")["log_liters_sold"]
                .count()
                .nlargest(n)
                .index.tolist()
            )

        self.n = len(self.selected_stores)
        return self