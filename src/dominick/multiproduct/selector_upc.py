import pandas as pd

class UPCSelector:
    """
    Selects UPCs for the multi-product pipeline.
    - If explicit UPCs are passed: use those (in that order).
    - If upcs=None: use the n with most observations in train.
    - If upcs=None and n=None: use ALL UPCs from the dataset.
    """

    def fit(
        self,
        train_df: pd.DataFrame,
        n: int | None = None,
        upcs: list | None = None,
    ) -> "UPCSelector":

        if upcs is not None:
            # Manual mode: explicit UPCs
            self.selected_upcs = list(upcs)

        elif n is None:
            # All mode: all UPCs from the dataset
            self.selected_upcs = train_df["upc_code"].unique().tolist()

        else:
            # Automatic mode: top-n by coverage
            self.selected_upcs = (
                train_df.groupby("upc_code")["log_liters_sold"]
                .count()
                .nlargest(n)
                .index.tolist()
            )

        self.n = len(self.selected_upcs)
        return self