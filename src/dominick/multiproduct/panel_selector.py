import pandas as pd
import numpy as np

class PanelSelector:
    """
    Jointly selects UPCs and stores for the multi-product pipeline.
    The goal is to build a dense panel in the (store, week) space,
    maximizing observation overlap across the chosen products.
    General logic:
      1. Define the store scope:
         - If `stores` is provided, the DataFrame is restricted to those stores.
         - Otherwise, all stores available in `df` are used.
      2. Select UPCs within that scope:
         - Manual mode:
             * If `upcs` is provided, exactly those codes are used.
         - All-UPCs mode:
             * If both `upcs` and `n_upcs` are None, all UPCs present
               in `df_scope` are selected.
         - Greedy mode (N UPCs):
             * If `n_upcs` is not None and `upcs` is None, for each UPC
               the set of (store_code, week_id) pairs where it appears
               is computed.
             * The UPC with the largest coverage (most pairs) is chosen first.
             * Then, iteratively, the next UPC that maximizes the size of
               the intersection with the current accumulated set is selected,
               updating that intersection at each step.
             * This yields a subset of `n_upcs` products sharing the
               densest possible (store, week) overlap.
      3. Select final stores:
         - `selected_stores` contains all stores that have at least one
           selected UPC (no completeness filter is applied at this stage).
         - Further filtering (e.g. minimum number of products per store)
           is handled downstream by `CompleteObservationFilter`.
    Attributes set after `fit`:
      - `selected_upcs`   : list of selected UPC codes.
      - `selected_stores` : list of stores containing at least one
                            selected UPC.
      - `n`               : number of selected UPCs.
    Public API:
      - `fit(df, n_upcs=None, upcs=None, stores=None) -> self`:
          Fits the selector to DataFrame `df` and stores the UPC and
          store selection as instance attributes.
    """

    def fit(
        self,
        df: pd.DataFrame,
        n_upcs: int | None = None,
        upcs: list | None = None,
        stores: list | None = None,
        n_time_bins: int = 5,
    ) -> "PanelSelector":
        """
        n_time_bins: number of chronological chunks used to score temporal
        balance. A candidate is scored by its WORST bin, not its total count,
        so products that go quiet for part of the horizon (e.g. discontinued
        SKUs) can no longer look as good as products that are steadily
        available throughout. Purely a scoring change -- the greedy structure
        (seed + iterative intersection) is unchanged.
        """
        # ── 1. Store scope ─────────────────────────────────────────
        if stores is not None:
            df_scope = df[df["store_code"].isin(stores)]
        else:
            df_scope = df

        # ── 2. UPC selection ────────────────────────────────────────
        if upcs is not None:
            self.selected_upcs = list(upcs)

        elif n_upcs is None:
            self.selected_upcs = df_scope["upc_code"].unique().tolist()

        else:
            df_scope = df_scope.copy()
            weeks_sorted = sorted(df_scope["week_id"].unique())
            # NEW: chronological bin index per week, used only for scoring.
            bin_of_week = {
                w: b
                for b, chunk in enumerate(np.array_split(weeks_sorted, n_time_bins))
                for w in chunk
            }
            df_scope["_time_bin"] = df_scope["week_id"].map(bin_of_week)

            # Same (store, week) sets as before...
            upc_storewks: dict = {
                upc: set(zip(g["store_code"], g["week_id"]))
                for upc, g in df_scope.groupby("upc_code")
            }
            # ...plus a per-bin breakdown, used only to penalize discontinuity.
            upc_bin_counts: dict = {
                upc: g.groupby("_time_bin").size().reindex(range(n_time_bins), fill_value=0)
                for upc, g in df_scope.groupby("upc_code")
            }

            def _worst_bin(counts) -> int:
                """Smallest per-bin count: a product empty in ANY bin scores 0
                here, regardless of how dense it is in the other bins."""
                return int(counts.min())

            # Seed: best worst-case temporal balance, not best raw total.
            first = max(upc_storewks, key=lambda u: _worst_bin(upc_bin_counts[u]))
            selected = [first]
            current_intersection = upc_storewks[first]

            for _ in range(n_upcs - 1):
                remaining = [u for u in upc_storewks if u not in selected]
                if not remaining:
                    break

                def _score(u):
                    inter = current_intersection & upc_storewks[u]
                    if not inter:
                        return 0
                    bins = pd.Series([bin_of_week[w] for _, w in inter])
                    per_bin = bins.value_counts().reindex(range(n_time_bins), fill_value=0)
                    return _worst_bin(per_bin)  # <- worst bin, not len(inter)

                best = max(remaining, key=_score)
                selected.append(best)
                current_intersection &= upc_storewks[best]

            self.selected_upcs = selected

        self.selected_stores = df[df["upc_code"].isin(self.selected_upcs)]["store_code"].unique().tolist()
        self.n = len(self.selected_upcs)
        return self