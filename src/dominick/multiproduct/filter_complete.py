import pandas as pd

class CompleteObservationFilter:
    """
    Filters rows (store, week) where all selected UPCs are present.
    Public API: run(df, selected_upcs) → filtered df
    """
    def run(self, df: pd.DataFrame, selected_upcs: list) -> pd.DataFrame:
        # 1. Stay only with the rows of the selected UPCs
        df = df[df["upc_code"].isin(selected_upcs)].copy()

        # 2. Keep (store, week) where all n UPCs are present
        counts = (
            df.groupby(["store_code", "week_id"])["upc_code"]
            .nunique()
        )
        complete = counts[counts == len(selected_upcs)].index
        df = df.set_index(["store_code", "week_id"])
        df = df.loc[df.index.isin(complete)].reset_index()
        return df