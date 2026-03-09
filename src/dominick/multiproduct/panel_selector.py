import pandas as pd


class PanelSelector:
    """
    Jointly selects UPCs and stores for the multi-product pipeline.

    Logic:
      1. Optionally restrict to specified stores (or use all).
      2. Within that store scope, select N UPCs with maximum joint
         (store, week) overlap using a greedy algorithm.
      3. selected_stores: all stores that have at least 1 of the
         selected UPCs (CompleteObservationFilter handles min_products).

    Public API: fit(df, n_upcs, upcs, stores) → self
    """

    def fit(
        self,
        df: pd.DataFrame,
        n_upcs: int | None = None,
        upcs: list | None = None,
        stores: list | None = None,
    ) -> "PanelSelector":

        # ── 1. Scope de tiendas ───────────────────────────────────────
        if stores is not None:
            df_scope = df[df["store_code"].isin(stores)]
        else:
            df_scope = df   # todas las tiendas

        # ── 2. Selección de UPCs ──────────────────────────────────────
        if upcs is not None:
            # Modo manual
            self.selected_upcs = list(upcs)

        elif n_upcs is None:
            # Todos los UPCs dentro del scope
            self.selected_upcs = df_scope["upc_code"].unique().tolist()

        else:
            # Greedy: maximiza intersección de (store, week) entre los n UPCs
            upc_storewks: dict = {
                upc: set(zip(g["store_code"], g["week_id"]))
                for upc, g in df_scope.groupby("upc_code")
            }

            first = max(upc_storewks, key=lambda u: len(upc_storewks[u]))
            selected = [first]
            current_intersection = upc_storewks[first]

            for _ in range(n_upcs - 1):
                remaining = [u for u in upc_storewks if u not in selected]
                if not remaining:
                    break
                best = max(
                    remaining,
                    key=lambda u: len(current_intersection & upc_storewks[u]),
                )
                selected.append(best)
                current_intersection &= upc_storewks[best]

            self.selected_upcs = selected

        # ── 3. Tiendas: todas las que tienen al menos 1 UPC seleccionado ──
        # (CompleteObservationFilter aplicará min_products después)
        self.selected_stores = (
            df[df["upc_code"].isin(self.selected_upcs)]["store_code"]
            .unique().tolist()
        )

        self.n = len(self.selected_upcs)
        return self