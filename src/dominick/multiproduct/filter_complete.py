import pandas as pd

class CompleteObservationFilter:
    """
    Filters UPCs and (store, week) rows for the multi-product pipeline.

    UPC-level filter: keeps only UPCs present in at least `min_coverage`
    fraction of (store, week) pairs in the dataset.

    Row-level filter: keeps (store, week) pairs where at least `min_products`
    of the selected UPCs are observed. Does NOT require full completeness.

    Public API: run(df, selected_upcs, selected_stores) → filtered df
    """

    def __init__(self, min_coverage: float = 0.5, min_products: int | None = None):
        """
        Args:
            min_coverage: minimum fraction of (store, week) pairs a UPC must
                          appear in to be kept (default: 0.5).
            min_products: minimum number of selected UPCs that must be present
                          in a (store, week) to keep that row. If None, uses
                          len(selected_upcs) (original strict behavior).
        """
        self.min_coverage = min_coverage
        self.min_products = min_products

    def run(
        self,
        df: pd.DataFrame,
        selected_upcs: list,
        selected_stores: list | None = None,
    ) -> pd.DataFrame:
        df = df[df["upc_code"].isin(selected_upcs)].copy()
        if selected_stores is not None:
            df = df[df["store_code"].isin(selected_stores)].copy()

        # ── UPC-level coverage filter ──────────────────────────────────────
        total_store_weeks = df[["store_code", "week_id"]].drop_duplicates().shape[0]
        upc_coverage = (
            df.groupby("upc_code")[["store_code", "week_id"]]
            .apply(lambda g: g.drop_duplicates().shape[0])
            / total_store_weeks
        )
        valid_upcs = upc_coverage[upc_coverage >= self.min_coverage].index.tolist()
        # Keep only UPCs that pass coverage AND were originally selected
        valid_upcs = [u for u in selected_upcs if u in valid_upcs]
        df = df[df["upc_code"].isin(valid_upcs)].copy()

        # ── Row-level filter ───────────────────────────────────────────────
        min_prod = self.min_products if self.min_products is not None else len(valid_upcs)
        counts = df.groupby(["store_code", "week_id"])["upc_code"].nunique()
        complete = counts[counts >= min_prod].index
        df = df.set_index(["store_code", "week_id"])
        df = df.loc[df.index.isin(complete)].reset_index()

        self.valid_upcs = valid_upcs
        return df