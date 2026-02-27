import pandas as pd

class CompleteObservationFilter:
    """
    Filters rows (store, week) where all selected UPCs are present.
    Public API: run(df, selected_upcs) → filtered df
    """
    def run(self, df: pd.DataFrame, selected_upcs: list, selected_stores: list | None = None) -> pd.DataFrame:

        df = df[df["upc_code"].isin(selected_upcs)].copy()
        if selected_stores is not None:
            df = df[df["store_code"].isin(selected_stores)].copy()

        counts = df.groupby(["store_code", "week_id"])["upc_code"].nunique()
        complete = counts[counts == len(selected_upcs)].index
        df = df.set_index(["store_code", "week_id"])
        df = df.loc[df.index.isin(complete)].reset_index()
        return df