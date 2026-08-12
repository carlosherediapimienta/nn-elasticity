import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import MLPConfig


class _TrainFitCategoryEncoder:
    """
    Factorizes a column using ONLY train values; maps val values to the
    same codes, with unseen categories mapped to a dedicated 'unknown'
    index. Avoids leakage (never fits on val) and avoids KeyErrors on
    categories that appear only in validation.
    """
    def fit(self, values: pd.Series) -> "_TrainFitCategoryEncoder":
        self.categories_ = pd.Index(sorted(values.unique()))
        self._map = {v: i for i, v in enumerate(self.categories_)}
        self.n_categories = len(self.categories_) + 1  # +1 reserved for "unknown"
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        unknown_idx = len(self.categories_)
        return values.map(self._map).fillna(unknown_idx).astype(np.int64).values


class DemandMLP(nn.Module):
    """
    Generic demand-first MLP: y_hat = f_theta(u, x).
      x = [log_p_i, log_p_j]  -- differentiated to obtain elasticities via autodiff
      u = [controls, store embedding, upc_i embedding, upc_j embedding]  -- context
    A SINGLE global model is trained on every (store, pair, upc_i, upc_j)
    row of the dyadic dataset at once (contrast with
    RegularizedElasticityPipeline, which fits one model per group). Store
    and product identity enter only as learned embeddings, replacing the
    per-group fixed effect / dummy.
    Deliberately excludes: splines, sparse attention, U_ij bilinear
    interaction, elasticity penalties/constraints, analytic derivatives.
    """

    def __init__(
        self,
        n_controls: int,
        n_stores: int,
        n_upcs: int,
        config: MLPConfig,
        d_store: int = 8,
        d_upc: int = 8,
    ):
        super().__init__()
        act_fn = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}[config.act]

        self.emb_store = nn.Embedding(n_stores, d_store)
        self.emb_upc = nn.Embedding(n_upcs, d_upc)  # shared vocabulary: upc_i and upc_j are the same kind of entity

        input_dim = 2 + n_controls + d_store + 2 * d_upc
        dims = [input_dim] + list(config.hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act_fn(), nn.Dropout(config.dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, controls, store_idx, upc_i_idx, upc_j_idx):
        # x: (B, 2) = standardized [log_p_i, log_p_j] -- must keep requires_grad
        # for the caller to compute elasticities via autodiff.
        e_store = self.emb_store(store_idx)
        e_i = self.emb_upc(upc_i_idx)
        e_j = self.emb_upc(upc_j_idx)
        inp = torch.cat([x, controls, e_store, e_i, e_j], dim=-1)
        return self.net(inp).squeeze(-1)


class DemandMLPPipeline:
    """
    Trains a SINGLE global DemandMLP pooling every (store, pair) row of the
    dyadic dataset at once. Elasticities are computed via autodiff on the
    validation set:
        own_elasticity   = d y_hat / d log_p_i
        cross_elasticity = d y_hat / d log_p_j
    (contrast with OLS/Ridge, whose elasticity is a FIXED coefficient per
    group; the MLP's elasticity is a point estimate that varies per
    observation -- non-linear, like ICDN's E matrix, but obtained by
    generic backward-mode autodiff instead of a closed-form spline formula).
    Public API: run(train_df, val_df) -> (metrics: dict, elasticities: pd.DataFrame)
    """

    def __init__(self, config: MLPConfig, device: str = "cpu", seed: int = 42):
        self.config = config
        self.device = device
        self.seed = seed

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
        torch.manual_seed(self.seed)
        cfg = self.config

        needed_cols = (
            ["store_code", "pair_id", "upc_i", "upc_j", "week_id", "log_v_i", "log_p_i", "log_p_j"]
            + cfg.control_cols
        )
        train_df = train_df[needed_cols].dropna().reset_index(drop=True)
        val_df = val_df[needed_cols].dropna().reset_index(drop=True)

        # ── Categorical encoders (fit on TRAIN only) ─────────────────────
        store_enc = _TrainFitCategoryEncoder().fit(train_df["store_code"])
        upc_enc = _TrainFitCategoryEncoder().fit(
            pd.concat([train_df["upc_i"], train_df["upc_j"]])
        )

        def _cat_tensors(df):
            return (
                torch.tensor(store_enc.transform(df["store_code"]), dtype=torch.long),
                torch.tensor(upc_enc.transform(df["upc_i"]), dtype=torch.long),
                torch.tensor(upc_enc.transform(df["upc_j"]), dtype=torch.long),
            )

        # ── Continuous features: standardize using TRAIN only ─────────────
        cont_cols = ["log_p_i", "log_p_j"] + cfg.control_cols
        mean = train_df[cont_cols].mean()
        std = train_df[cont_cols].std().replace(0, 1.0)

        def _cont_tensor(df):
            z = (df[cont_cols] - mean) / std
            return torch.tensor(z.values, dtype=torch.float32)

        Xtr_cont = _cont_tensor(train_df).to(self.device)
        Xval_cont = _cont_tensor(val_df).to(self.device)
        store_tr, upc_i_tr, upc_j_tr = (t.to(self.device) for t in _cat_tensors(train_df))
        store_val, upc_i_val, upc_j_val = (t.to(self.device) for t in _cat_tensors(val_df))
        ytr = torch.tensor(train_df["log_v_i"].values, dtype=torch.float32, device=self.device)
        yval = torch.tensor(val_df["log_v_i"].values, dtype=torch.float32, device=self.device)

        model = DemandMLP(
            n_controls=len(cfg.control_cols),
            n_stores=store_enc.n_categories,
            n_upcs=upc_enc.n_categories,
            config=cfg,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        huber = nn.HuberLoss(delta=cfg.huber_delta)

        best_val_loss, no_improve, best_state = float("inf"), 0, None
        n_train = len(train_df)

        for _epoch in range(cfg.n_epochs):
            model.train()
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                optimizer.zero_grad()
                y_hat = model(
                    Xtr_cont[idx, :2], Xtr_cont[idx, 2:],
                    store_tr[idx], upc_i_tr[idx], upc_j_tr[idx],
                )
                loss = huber(y_hat, ytr[idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_hat_val = model(Xval_cont[:, :2], Xval_cont[:, 2:], store_val, upc_i_val, upc_j_val)
                val_loss = huber(y_hat_val, yval).item()

            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= cfg.es_patience:
                break

        model.load_state_dict(best_state)
        model.eval()

        # ── Metrics on val ────────────────────────────────────────────────
        with torch.no_grad():
            y_hat_val = model(Xval_cont[:, :2], Xval_cont[:, 2:], store_val, upc_i_val, upc_j_val)
        residuals = (yval - y_hat_val).cpu().numpy()
        mae_val = float(np.mean(np.abs(residuals)))
        rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        yval_np = yval.cpu().numpy()
        ss_tot = float(np.sum((yval_np - yval_np.mean()) ** 2))
        r2_val = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

        # ── Elasticities via autodiff (own = dy/d log_p_i, cross = dy/d log_p_j) ──
        x_val_grad = Xval_cont[:, :2].clone().requires_grad_(True)
        y_hat_grad = model(x_val_grad, Xval_cont[:, 2:], store_val, upc_i_val, upc_j_val)
        grad_x, = torch.autograd.grad(y_hat_grad.sum(), x_val_grad, create_graph=False)
        # x was standardized: z = (log_p - mean) / std  =>  dy/d(log_p) = (dy/dz) / std
        own_elasticity = (grad_x[:, 0] / std["log_p_i"]).detach().cpu().numpy()
        cross_elasticity = (grad_x[:, 1] / std["log_p_j"]).detach().cpu().numpy()

        elasticities = pd.DataFrame({
            "store_code": val_df["store_code"].values,
            "pair_id": val_df["pair_id"].values,
            "upc_i": val_df["upc_i"].values,
            "upc_j": val_df["upc_j"].values,
            "week_id": val_df["week_id"].values,
            "own_elasticity": own_elasticity,
            "cross_elasticity": cross_elasticity,
            "y_true_i": val_df["log_v_i"].values,
            "y_hat_i": y_hat_grad.detach().cpu().numpy(),
        })

        metrics = {
            "mae_val": mae_val, "rmse_val": rmse_val, "r2_val": r2_val,
            "best_val_loss": best_val_loss,
        }
        return metrics, elasticities