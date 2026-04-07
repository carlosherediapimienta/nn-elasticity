import pandas as pd


class CrossElasticitySymmetrizer:

    def symmetrize(
        self,
        df: pd.DataFrame,
        run_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Average cross-elasticities from both directions (i to j and j to i) into 
        a single canonical estimate. The pairwise OLS produces two directed estimates 
        for each store x pair: the effect of upc_j's price on upc_i's demand, and vice versa.
        This method averages them into one symmetric estimate per canonical pair (upc_a, upc_b),
        where upc_a < upc_b by construction.
        Args:
            df: Directional results pre-filtered to status == "ok".
            run_col: Column identifying the evaluation run (e.g. "fold" or
                "bootstrap_run"). When provided, symmetrization is performed
                independently within each run, so the output retains one row
                per (store, pair, run) rather than collapsing across runs.
        """
        if df.empty:
            return pd.DataFrame()

        # Create the canonical pair (upc_a, upc_b)
        tmp = df.copy()
        tmp["upc_a"] = tmp[["upc_i", "upc_j"]].min(axis=1)
        tmp["upc_b"] = tmp[["upc_i", "upc_j"]].max(axis=1)

        # Group by the canonical pair and the run column
        group_cols = ["store_code", "pair_id", "upc_a", "upc_b"]
        if run_col is not None:
            group_cols.append(run_col)

        # Aggregate the cross-elasticities from both directions into a single symmetric estimate
        return (
            tmp
            .groupby(group_cols, as_index=False)
            .agg(
                cross_elasticity_sym=("cross_elasticity", "mean"),
                n_directions=("cross_elasticity", "count"),
                avg_p_value=("cross_elasticity_p_value", "mean"),
                mae_val_mean=("mae_val", "mean"),
                rmse_val_mean=("rmse_val", "mean"),
                r2_val_mean=("r2_val", "mean"),
            )
        )