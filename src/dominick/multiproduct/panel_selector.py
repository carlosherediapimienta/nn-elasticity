import pandas as pd

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
    ) -> "PanelSelector":

        # ── 1. Store scope ───────────────────────────────────────
        if stores is not None:
            df_scope = df[df["store_code"].isin(stores)]
        else:
            df_scope = df   # all stores

        # ── 2. UPC selection ──────────────────────────────────────
        if upcs is not None:
            # Manual mode
            self.selected_upcs = list(upcs)

        elif n_upcs is None:
            # All UPCs within the scope
            self.selected_upcs = df_scope["upc_code"].unique().tolist()

        else:
            # Greedy: maximize intersection of (store, week) between the n UPCs
            # Compute the set of (store, week) pairs for each UPC
            # {
            #    upc_1: {(storeA, week1), (storeA, week2), (storeB, week1), ...},
            #    upc_2: {(storeA, week1), (storeC, week5), ...},
            #    ...
            # }
            upc_storewks: dict = {
                upc: set(zip(g["store_code"], g["week_id"]))
                for upc, g in df_scope.groupby("upc_code")
            }

            # Select the UPC with the largest coverage
            # and add it to the selected list
            first = max(upc_storewks, key=lambda u: len(upc_storewks[u]))
            selected = [first]
            current_intersection = upc_storewks[first]

            # Select the next UPC with the largest coverage
            # and add it to the selected list
            # and update the current intersection
            for _ in range(n_upcs - 1):
                remaining = [u for u in upc_storewks if u not in selected] # Remaining UPCs
                if not remaining: # If there are no remaining UPCs, break the loop
                    break
                best = max(remaining, key=lambda u: len(current_intersection & upc_storewks[u])) # Best UPC to add
                selected.append(best) # Add the best UPC to the selected list
                current_intersection &= upc_storewks[best] # Update the current intersection

            self.selected_upcs = selected

        # Stores: all stores that have at least 1 selected UPC
        # (CompleteObservationFilter will apply min_products after)
        self.selected_stores = df[df["upc_code"].isin(self.selected_upcs)]["store_code"].unique().tolist()
        self.n = len(self.selected_upcs) # Number of selected UPCs
        return self