import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import MLPConfig

class MultiproductMLP(nn.Module):
    """Dense MLP: z = [prices, shared, vec(product feats), brand/style/store emb] → n log-demands.

    The observation mask is not an input. Elasticities come from autodiff on `prices`.
    """

    def __init__(
        self,
        n: int,
        n_stores: int,
        n_brands: int,
        n_styles: int,
        n_product_feats: int,
        n_shared: int = 8,
        hidden: tuple = (64, 32),
        act: str = "gelu",
        dropout: float = 0.0,
        d_store: int = 16,
        d_brand: int = 8,
        d_style: int = 8,
    ):
        super().__init__()
        self.n = n
        self.emb_store = nn.Embedding(n_stores, d_store)
        self.emb_brand = nn.Embedding(n_brands + 1, d_brand, padding_idx=0)
        self.emb_style = nn.Embedding(n_styles + 1, d_style, padding_idx=0)

        act_fn = {
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
            "softplus": nn.Softplus,
        }[act]
        in_dim = (
            n
            + n_shared
            + n * n_product_feats
            + n * d_brand
            + n * d_style
            + d_store
        )
        dims = [in_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act_fn(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n))
        self.net = nn.Sequential(*layers)

    def _context(self, batch: dict) -> torch.Tensor:
        """Non-price features. Shape (B, n_shared + n·F + n·d_brand + n·d_style)."""
        B = batch["ids"].shape[0]
        n = self.n
        e_brand = self.emb_brand(batch["per_prod_cat"][:, :, 0]).reshape(B, n * self.emb_brand.embedding_dim)
        e_style = self.emb_style(batch["per_prod_cat"][:, :, 1]).reshape(B, n * self.emb_style.embedding_dim)
        xp = torch.cat(
            [batch["per_prod_float"], batch["availability"].unsqueeze(-1)],
            dim=-1,
        ).reshape(B, -1)
        return torch.cat([batch["time_feats"], batch["promo_feats"], xp, e_brand, e_style], dim=-1)

    def forward(self, prices: torch.Tensor, batch: dict) -> torch.Tensor:
        """
        Args:
            prices: (B, n) log-prices — keep requires_grad for elasticities.
            batch:  MultiProductDataset batch (ids, time_feats, promo_feats, ...).
        Returns:
            y_hat: (B, n)
        """
        avail = batch["availability"]
        prices = prices * avail                   
        e_store = self.emb_store(batch["ids"][:, 0])
        z = torch.cat([prices, self._context(batch), e_store], dim=-1)
        return self.net(z)



class DemandMLPPipeline:
    """Fit a MultiproductMLP on ICDN-style DataLoaders. Elasticities come in a later step."""

    def __init__(
        self,
        config: MLPConfig,
        n: int,
        n_stores: int,
        n_brands: int,
        n_styles: int,
        n_product_feats: int,
        device: str | None = None,
        seed: int = 42,
    ):
        self.config = config
        self.n = n
        self.n_stores = n_stores
        self.n_brands = n_brands
        self.n_styles = n_styles
        self.n_product_feats = n_product_feats
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.model: MultiproductMLP | None = None
        self.best_val_loss: float | None = None

    def _build_model(self) -> MultiproductMLP:
        cfg = self.config
        return MultiproductMLP(
            n=self.n,
            n_stores=self.n_stores,
            n_brands=self.n_brands,
            n_styles=self.n_styles,
            n_product_feats=self.n_product_feats,
            hidden=cfg.hidden,
            act=cfg.act,
            dropout=cfg.dropout,
            d_store=cfg.d_store,
        ).to(self.device)

    @staticmethod
    def _masked_huber(huber: nn.Module, y_hat, y, w):
        # w = obs_mask. Same as ICDN: Huber only on observed cells.
        return (huber(y_hat, y) * w).sum() / w.sum().clamp_min(1.0)

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self, train_loader, val_loader) -> "DemandMLPPipeline":
        self._seed_everything(self.seed)
        cfg = self.config
        model = self._build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=20, min_lr=1e-5,
        )
        huber = nn.HuberLoss(delta=cfg.huber_delta, reduction="none")

        best_val, no_improve, best_state = float("inf"), 0, None

        for epoch in range(1, cfg.n_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = self._move(batch)
                opt.zero_grad()
                y_hat = model(batch["prices"], batch)
                loss = self._masked_huber(huber, y_hat, batch["demands"], batch["obs_mask"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            val_sum, val_den = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = self._move(batch)
                    w = batch["obs_mask"]
                    y_hat = model(batch["prices"], batch)
                    val_sum += float(self._masked_huber(huber, y_hat, batch["demands"], w).item()) * float(w.sum())
                    val_den += float(w.sum())
            val_loss = val_sum / max(val_den, 1.0)
            prev_lr = opt.param_groups[0]["lr"]
            sch.step(val_loss)
            if opt.param_groups[0]["lr"] < prev_lr:
                no_improve = 0

            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= cfg.es_patience:
                    print(f"early stop at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.best_val_loss = best_val
        return self

    def metrics(self, loader) -> dict:
        """MAE / RMSE / R² on observed cells only. No elasticities yet."""
        if self.model is None:
            raise RuntimeError("fit() first")
        self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                batch = self._move(batch)
                y_hat = self.model(batch["prices"], batch)
                m = batch["obs_mask"].bool()
                ys.append(batch["demands"][m].cpu())
                ps.append(y_hat[m].cpu())
        y = torch.cat(ys).numpy()
        p = torch.cat(ps).numpy()
        resid = y - p
        ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
        return {
            "mae_val": float(np.mean(np.abs(resid))) if y.size else np.nan,
            "rmse_val": float(np.sqrt(np.mean(resid ** 2))) if y.size else np.nan,
            "r2_val": np.nan if ss_tot == 0 else float(1.0 - np.sum(resid ** 2) / ss_tot),
            "best_val_loss": self.best_val_loss,
            "n_cells": int(y.size),
        }
        
    def elasticity_score(
        self,
        loader,
        own_min: float = -5.0,
        own_max: float = 0.0,
        cross_min: float = -1.0,
        cross_max: float = 1.0,
        beta_eda: float = -2.0,
    ) -> dict:
        """ICDN `compute_elasticity_score` on the dense Jacobian.

        Own = diagonal on observed i. Cross = off-diagonal on observed (i, j).
        No neighbor graph: the MLP has none.
        """
        if self.model is None:
            raise RuntimeError("fit() first")
        model = self.model
        model.eval()
        n = self.n
        all_own, all_cross = [], []

        for batch in loader:
            batch = self._move(batch)
            prices = batch["prices"].detach().clone().requires_grad_(True)
            y_hat = model(prices, batch)
            grads = []
            for i in range(n):
                g, = torch.autograd.grad(
                    y_hat[:, i].sum(), prices, retain_graph=True,
                )
                grads.append(g)
            E = torch.stack(grads, dim=1)  # (B, n, n)
            m = batch["obs_mask"].bool() & batch["price_observed"].bool()
            eye = torch.eye(n, dtype=torch.bool, device=E.device)

            all_own.append(torch.diagonal(E, dim1=1, dim2=2)[m].detach().cpu())
            pair = m.unsqueeze(2) & m.unsqueeze(1) & ~eye
            all_cross.append(E[pair].detach().cpu())

        own = torch.cat(all_own).numpy() if all_own else np.array([])
        cross = torch.cat(all_cross).numpy() if all_cross else np.array([])

        if own.size:
            own_in_range = float(((own >= own_min) & (own <= own_max)).mean())
            median_own = float(np.median(own))
            deviation = max(0.0, abs(median_own - beta_eda) - 0.3)
            prior_penalty = min(deviation / abs(beta_eda), 1.0)
        else:
            own_in_range, median_own, prior_penalty = 0.0, np.nan, 1.0
        own_score = own_in_range * (1.0 - prior_penalty)

        if cross.size:
            cross_in_range = float(((cross >= cross_min) & (cross <= cross_max)).mean())
            median_cross = float(np.median(cross))
        else:
            cross_in_range, median_cross = 1.0, np.nan

        return {
            "elast_score": float(0.7 * own_score + 0.3 * cross_in_range),
            "own_score": float(own_score),
            "own_in_range": float(own_in_range),
            "own_elasticity_median": median_own,
            "cross_in_range": float(cross_in_range),
            "cross_elasticity_median": median_cross,
        }
    

    def elasticities(self, loader, store_cats, upc_names, week_cats=None):
        """Per-cell Jacobian: one row per observed (i, j). Own = diagonal, cross = off-diagonal."""
        if self.model is None:
            raise RuntimeError("fit() first")
        model = self.model
        model.eval()
        n = self.n
        upc_names = np.asarray(upc_names)
        store_cats = np.asarray(store_cats)
        week_cats = None if week_cats is None else np.asarray(week_cats)
        rows = []

        for batch in loader:
            batch = self._move(batch)
            prices = batch["prices"].detach().clone().requires_grad_(True)
            y_hat = model(prices, batch)
            grads = []
            for i in range(n):
                g, = torch.autograd.grad(
                    y_hat[:, i].sum(), prices, retain_graph=True,
                )
                grads.append(g)
            E = torch.stack(grads, dim=1).detach().cpu().numpy()  # (B, n, n)

            mask = batch["obs_mask"].bool().cpu().numpy()
            pobs = batch["price_observed"].cpu().numpy().astype(bool)
            store_idx = batch["ids"][:, 0].cpu().numpy()
            week_idx = batch["ids"][:, 1].cpu().numpy()
            y_true = batch["demands"].detach().cpu().numpy()
            y_hat_np = y_hat.detach().cpu().numpy()

            for b in range(E.shape[0]):
                sc = store_cats[store_idx[b]]
                wc = None if week_cats is None else week_cats[week_idx[b]]
                for i in range(n):
                    if not mask[b, i] or not pobs[b, i]:
                        continue
                    for j in range(n):
                        if not mask[b, j] or not pobs[b, j]:
                            continue
                        row = {
                            "store_code": sc,
                            "upc_i": upc_names[i],
                            "upc_j": upc_names[j],
                            "type": "own" if i == j else "cross",
                            "E": float(E[b, i, j]),
                            "y_true_i": float(y_true[b, i]),
                            "y_hat_i": float(y_hat_np[b, i]),
                        }
                        if wc is not None:
                            row["week_id"] = wc
                        rows.append(row)

        return pd.DataFrame(rows)

    def evaluate(self, loader, store_cats, upc_names, week_cats=None):
        return self.metrics(loader), self.elasticities(
            loader, store_cats, upc_names, week_cats=week_cats,
        )