import pandas as pd


class StoreWeekCoverageAnalyzer:
    """
    Analyzes store-week coverage:
    - Global coverage (observed store-weeks vs possible)
    - Distribution of number of weeks per store
    Public API: run().
    """

    def run(
        self,
        df: pd.DataFrame,
        store_col: str = "store_code",
        week_col: str = "week_id",
        min_weeks_for_good: int = 150,
    ) -> dict:
        """
        Args:
            df: DataFrame with grain (store, upc, week) without duplicates.
            store_col: store column.
            week_col: week column.
            min_weeks_for_good: threshold of weeks to mark stores with low coverage.

        Returns:
            dict with:
                - n_stores: nº de stores distintos
                - n_weeks_global: number of distinct weeks in the dataset
                - n_store_week_possible: possible store-weeks combinations
                                         = n_stores × n_weeks_global
                - n_store_week_observed: observed store-weeks combinations
                                         (unique store-week pairs)
                - store_week_density: density in [0,1]
                - store_week_density_pct: density in %
                - n_missing_store_weeks: number of missing store-weeks
                - pct_missing_store_weeks: % of missing store-weeks
                - min_weeks_for_good: threshold used
                - n_low_coverage_stores: number of stores with weeks < threshold
                - low_coverage_stores: list of IDs of those stores
                - per_store_coverage: DataFrame con:
                    * store_col: store column
                    * n_weeks_store: number of weeks per store
                    * coverage_pct: coverage percentage
                    * is_low_coverage: bool, True if the store has low coverage
        """
        for col in [store_col, week_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # Stats globales
        n_stores = int(df[store_col].nunique(dropna=True))
        n_weeks_global = int(df[week_col].nunique(dropna=True))

        n_store_week_possible = n_stores * n_weeks_global

        # Pairs store-week observados (independientemente de UPCs)
        n_store_week_observed = int(
            df[[store_col, week_col]].drop_duplicates().shape[0]
        )

        if n_store_week_possible > 0:
            store_week_density = n_store_week_observed / n_store_week_possible
        else:
            store_week_density = 0.0

        store_week_density_pct = round(store_week_density * 100, 4)

        n_missing_store_weeks = max(n_store_week_possible - n_store_week_observed, 0)
        pct_missing_store_weeks = (
            round(n_missing_store_weeks / n_store_week_possible * 100, 4)
            if n_store_week_possible > 0
            else 0.0
        )

        # Distribution of weeks by store
        per_store = (
            df[[store_col, week_col]]
            .drop_duplicates()
            .groupby(store_col)[week_col]
            .nunique()
            .rename("n_weeks_store")
            .reset_index()
        )

        if n_weeks_global > 0:
            per_store["coverage_pct"] = (
                per_store["n_weeks_store"] / n_weeks_global * 100
            )
        else:
            per_store["coverage_pct"] = 0.0

        per_store["is_low_coverage"] = per_store["n_weeks_store"] < min_weeks_for_good

        low_cov = per_store[per_store["is_low_coverage"]]
        n_low_coverage_stores = int(low_cov.shape[0])
        low_coverage_stores = low_cov[store_col].tolist()

        return {
            "n_stores": n_stores,
            "n_weeks_global": n_weeks_global,
            "n_store_week_possible": n_store_week_possible,
            "n_store_week_observed": n_store_week_observed,
            "store_week_density": store_week_density,
            "store_week_density_pct": store_week_density_pct,
            "n_missing_store_weeks": n_missing_store_weeks,
            "pct_missing_store_weeks": pct_missing_store_weeks,
            "min_weeks_for_good": min_weeks_for_good,
            "n_low_coverage_stores": n_low_coverage_stores,
            "low_coverage_stores": low_coverage_stores,
            "per_store_coverage": per_store,
            "store_col": store_col,
            "week_col": week_col,
        }