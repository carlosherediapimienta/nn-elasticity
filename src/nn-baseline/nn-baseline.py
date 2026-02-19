# multi_sku_elasticity_baselines.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

# PyTorch >= 2.0 recomendado (torch.func)
from torch.func import vmap, jacrev

# -------------------------
# Baseline 1: E constante (full matrix, incluye cross-effects)
# -------------------------
class ConstantMatrixElasticity(nn.Module):
    """
    E(x,c) = W (constante), W in R^{n x n}
    Integrable: y(x) = y0 + W (x - x0)
    """
    def __init__(self, n: int, init_scale: float = 1e-2):
        super().__init__()
        self.n = n
        W = init_scale * torch.randn(n, n)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        return self.W.unsqueeze(0).expand(B, self.n, self.n)


def closed_form_y_constant_matrix(W: torch.Tensor, x: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    # y = y0 + W (x-x0)
    dx = x - x0
    return y0 + torch.einsum("ij,bj->bi", W, dx)


# -------------------------
# Baseline 2: Elasticity-first NN: E_phi(x,c)
# -------------------------
class ElasticityFieldMLP(nn.Module):
    """
    Aprende E_phi(x,c) in R^{n x n}.
    Usa activaciones suaves (tanh/softplus) para derivadas.
    """
    def __init__(self, n: int, d_context: int = 0, hidden: int = 256, depth: int = 3, act: str = "tanh"):
        super().__init__()
        self.n = n
        self.d_context = d_context

        if act.lower() == "tanh":
            activation = nn.Tanh()
        elif act.lower() == "softplus":
            activation = nn.Softplus()
        else:
            # ReLU no es C^2 -> peor para chequear/castigar parciales mixtas
            activation = nn.ReLU()

        din = n + d_context
        layers = [nn.Linear(din, hidden), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation]
        layers += [nn.Linear(hidden, n * n)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.d_context > 0:
            if c is None:
                raise ValueError("d_context>0 pero c=None")
            inp = torch.cat([x, c], dim=-1)
        else:
            inp = x
        out = self.net(inp)                         # (B, n*n)
        return out.view(x.shape[0], self.n, self.n) # (B, n, n)


# -------------------------
# Reconstrucción: camino coordenado con Euler
#   y_{t+1} = y_t + E(x_t,c) Δx_t
# -------------------------
def integrate_coordwise_euler(
    E_model: nn.Module,
    x0: torch.Tensor,        # (B,n)
    y0: torch.Tensor,        # (B,n)
    xT: torch.Tensor,        # (B,n)
    c: Optional[torch.Tensor] = None,
    order: Optional[Sequence[int]] = None,
    steps_per_dim: int = 16,
) -> torch.Tensor:
    B, n = x0.shape
    if order is None:
        order = list(range(n))
    if sorted(order) != list(range(n)):
        raise ValueError("order debe ser una permutación de [0..n-1]")

    x = x0.clone()
    y = y0.clone()

    for j in order:
        total = (xT[:, j] - x0[:, j])          # (B,)
        step = total / float(steps_per_dim)   # (B,)
        for _ in range(steps_per_dim):
            dx = torch.zeros_like(x)
            dx[:, j] = step
            E = E_model(x, c)                 # (B,n,n)
            y = y + torch.einsum("bij,bj->bi", E, dx)
            x[:, j] = x[:, j] + step
    return y


def path_dependence_gap(
    E_model: nn.Module,
    x0: torch.Tensor,
    y0: torch.Tensor,
    xT: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    steps_per_dim: int = 16,
) -> torch.Tensor:
    """
    Mide dependencia del orden (camino):
    gap = ||y_hat(order A) - y_hat(order B)||.
    """
    n = x0.shape[1]
    ya = integrate_coordwise_euler(E_model, x0, y0, xT, c=c, order=list(range(n)), steps_per_dim=steps_per_dim)
    yb = integrate_coordwise_euler(E_model, x0, y0, xT, c=c, order=list(reversed(range(n))), steps_per_dim=steps_per_dim)
    return torch.linalg.norm(ya - yb, dim=-1).mean()


# -------------------------
# Closure residuals: c^i_{jk} = ∂_k E_{i,j} - ∂_j E_{i,k}
# y penalización sum_{i,j<k} c^2
# -------------------------
@dataclass
class ClosureOut:
    penalty: torch.Tensor            # escalar
    per_sample: torch.Tensor         # (B,)
    residuals_upper: torch.Tensor    # (B, n, n_pairs)


def closure_penalty(
    E_model: nn.Module,
    x: torch.Tensor,                 # (B,n)
    c: Optional[torch.Tensor] = None,
    pair_subsample: Optional[int] = None,
) -> ClosureOut:
    """
    Calcula closure residuals para j<k. Coste ~ O(B * n^3).
    Para n grande, usa pair_subsample para muestrear pares (j,k).
    """
    B, n = x.shape
    device = x.device

    jk = torch.triu_indices(n, n, offset=1, device=device)  # (2, n_pairs)
    if pair_subsample is not None and pair_subsample < jk.shape[1]:
        idx = torch.randperm(jk.shape[1], device=device)[:pair_subsample]
        jk = jk[:, idx]
    n_pairs = jk.shape[1]

    def E_single(x1: torch.Tensor, c1: Optional[torch.Tensor]) -> torch.Tensor:
        xs = x1.unsqueeze(0)
        cs = None if c1 is None else c1.unsqueeze(0)
        return E_model(xs, cs)[0]   # (n,n)

    # dE_dx[b, i, j, k] = ∂ E_{i,j} / ∂ x_k
    if c is None:
        dE_dx = vmap(jacrev(lambda x1: E_single(x1, None)))(x.detach().requires_grad_(True))
    else:
        dE_dx = vmap(jacrev(lambda x1, c1: E_single(x1, c1), argnums=0))(
            x.detach().requires_grad_(True), c.detach()
        )

    # res[b,i,j,k] = ∂_k E_{i,j} - ∂_j E_{i,k}
    res = dE_dx - dE_dx.permute(0, 1, 3, 2)

    residuals_upper = res[:, :, jk[0], jk[1]]        # (B, n, n_pairs)
    per_sample = (residuals_upper ** 2).sum(dim=(1, 2))  # (B,)
    penalty = per_sample.mean()

    return ClosureOut(penalty=penalty, per_sample=per_sample, residuals_upper=residuals_upper)


# -------------------------
# Baseline 3 (recomendado como control): Potential-first y=g(x,c), E=J_x g
# Integrable by construction si g es C^2
# -------------------------
class PotentialMLP(nn.Module):
    def __init__(self, n: int, d_context: int = 0, hidden: int = 256, depth: int = 3, act: str = "tanh"):
        super().__init__()
        self.n = n
        self.d_context = d_context

        if act.lower() == "tanh":
            activation = nn.Tanh()
        elif act.lower() == "softplus":
            activation = nn.Softplus()
        else:
            activation = nn.ReLU()

        din = n + d_context
        layers = [nn.Linear(din, hidden), activation]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), activation]
        layers += [nn.Linear(hidden, n)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        inp = torch.cat([x, c], dim=-1) if (self.d_context > 0) else x
        return self.net(inp)   # (B,n)


def jacobian_elasticity_from_potential(g_model: nn.Module, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    E(x,c) = J_x g(x,c)  (B,n,n)
    """
    def g_single(x1: torch.Tensor, c1: Optional[torch.Tensor]) -> torch.Tensor:
        xs = x1.unsqueeze(0)
        cs = None if c1 is None else c1.unsqueeze(0)
        return g_model(xs, cs)[0]  # (n,)

    if c is None:
        J = vmap(jacrev(lambda x1: g_single(x1, None)))(x.detach().requires_grad_(True))
    else:
        J = vmap(jacrev(lambda x1, c1: g_single(x1, c1), argnums=0))(
            x.detach().requires_grad_(True), c.detach()
        )
    return J


# -------------------------
# Ejemplo rápido
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, n, d = 8, 6, 4
    x0 = torch.zeros(B, n)
    y0 = torch.zeros(B, n)
    xT = 0.2 * torch.randn(B, n)
    c  = torch.randn(B, d)

    # Elasticity-first
    Ephi = ElasticityFieldMLP(n=n, d_context=d, hidden=128, depth=3, act="tanh")
    y_hat = integrate_coordwise_euler(Ephi, x0, y0, xT, c=c, steps_per_dim=32)
    gap = path_dependence_gap(Ephi, x0, y0, xT, c=c, steps_per_dim=32)
    clo = closure_penalty(Ephi, xT, c=c, pair_subsample=20)

    print("Elasticity-first y_hat:", y_hat.shape)
    print("Path dependence gap:", float(gap))
    print("Closure penalty:", float(clo.penalty))

    # Potential-first (control)
    g = PotentialMLP(n=n, d_context=d, hidden=128, depth=3, act="tanh")
    E_from_g = jacobian_elasticity_from_potential(g, xT, c=c)
    # Si quieres, también puedes pasar E_from_g por closure_penalty implementando un wrapper E_model.
    print("Potential-first E:", E_from_g.shape)
