import pandas as pd

class CompleteObservationFilter:
    """
    Applies a two-stage filter to build a clean, dense panel for the
    multi-product pipeline.
    Stage 1 — UPC-level coverage filter:
        Computes, for each UPC, the fraction of unique (store, week) pairs
        in the dataset where it is observed. Only UPCs that meet or exceed
        `min_coverage` are retained. This removes sparse or intermittent
        products that would create too many missing entries in the panel.
    Stage 2 — Row-level density filter:
        Among the surviving UPCs, keeps only (store, week) pairs where
        at least `min_products` distinct UPCs are observed. This ensures
        that every row in the final panel has a minimum number of products
        present, without requiring full completeness across all UPCs.
        If `min_products` is None, the filter defaults to requiring ALL
        surviving UPCs to be present (strict completeness).
    Attributes set after `run`:
        - `valid_upcs`: list of UPCs that passed the coverage filter,
          preserving the original order from `selected_upcs`.
    Public API:
        - `__init__(min_coverage=0.5, min_products=None)`
        - `run(df, selected_upcs, selected_stores=None) → filtered df`
    Example:
        min_coverage=0.5, min_products=3 means:
          - drop any UPC present in fewer than 50% of (store, week) pairs,
          - then drop any (store, week) row where fewer than 3 UPCs appear.
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
        # Keep only the selected UPCs
        df = df[df["upc_code"].isin(selected_upcs)].copy()
        # Keep only the selected stores
        if selected_stores is not None:
            df = df[df["store_code"].isin(selected_stores)].copy()
        # UPC-level coverage filter
        total_store_weeks = df[["store_code", "week_id"]].drop_duplicates().shape[0]

        # Compute the coverage of each UPC
        # Interpretation: fraction of (store, week) pairs that the UPC is present in
        # over the total number of (store, week) pairs
        # Example: upc_coverage[upc] = 1.0 -> the UPC is present in all (store, week) pairs in the panel.
        # upc_coverage[upc] = 0.25 -> the UPC is present in 25% of the (store, week) pairs in the panel.
        upc_coverage = (
            df.groupby("upc_code")[["store_code", "week_id"]]
            .apply(lambda g: g.drop_duplicates().shape[0])
            / total_store_weeks
        )
        valid_upcs = upc_coverage[upc_coverage >= self.min_coverage].index.tolist() # Keep only UPCs that pass coverage
        valid_upcs = [u for u in selected_upcs if u in valid_upcs] # Keep only UPCs that were originally selected
        df = df[df["upc_code"].isin(valid_upcs)].copy() # Keep only the selected UPCs

        # Row-level filter: keep only (store, week) pairs where at least `min_products` of the selected UPCs are observed
        min_prod = self.min_products if self.min_products is not None else len(valid_upcs)
        counts = df.groupby(["store_code", "week_id"])["upc_code"].nunique() # Count the number of unique UPCs per (store, week) pair
        complete = counts[counts >= min_prod].index # Keep only (store, week) pairs where at least `min_products` of the selected UPCs are observed
        df = df.set_index(["store_code", "week_id"])
        df = df.loc[df.index.isin(complete)].reset_index() # Keep only the selected (store, week) pairs

        self.valid_upcs = valid_upcs # Store the valid UPCs
        return df