import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.func import vmap, jacrev


@dataclass
class CrossDerivativesOut:
    """Result of the cross-derivatives symmetry diagnostic."""
    penalty: torch.Tensor          # scalar: mean of ||J - J^T||²
    per_sample: torch.Tensor       # (B,)
    residuals_upper: torch.Tensor  # (B, n_pairs): upper triangle of J - J^T


class CrossDerivativesDiagnostics:
    """
    Certifies Slutsky symmetry of the demand Jacobian:
      S_{ij} = ∂y_i/∂x_j - ∂y_j/∂x_i  →  debe ser 0.

    Para el ICDN extendido con matriz A simétrica:
      J_{ij} = A_{ij}(c) = A_{ji}(c)  →  J = J^T exactamente por construcción.

    Penalty = mean_B( Σ_{i<j} S_{ij}² )  →  debe ser 0.0 para ICDN.

    Public API: run()
    """

    def run(self, model: nn.Module, batch: dict) -> CrossDerivativesOut:
        """
        Args:
            model: ICDN instance
            batch: dict con todos los tensores del batch

        Returns:
            CrossDerivativesOut(penalty, per_sample, residuals_upper)
        """
        model.eval()
        device = next(model.parameters()).device
        n = model.n

        with torch.no_grad():
            c = model.context_builder(batch)  # (B, d)

        x = torch.stack(
            [batch[f"log_price_{i}"] for i in range(n)], dim=1
        ).to(device)  # (B, n)

        def demand_fn(x1: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
            """(n,), (d,) -> (n,)"""
            x_b = x1.unsqueeze(0)
            c_b = c1.unsqueeze(0)
            Bx_list, dBx_list = [], []
            for i, spline in enumerate(model.price_splines):
                Bx_i, dBx_i, _ = spline(x_b[:, i])
                Bx_list.append(Bx_i)
                dBx_list.append(dBx_i)
            Bx  = torch.stack(Bx_list,  dim=1)
            dBx = torch.stack(dBx_list, dim=1)
            y_hat, _, _ = model.head.run(c_b, x_b, Bx, dBx)
            return y_hat[0]  # (n,)

        # J_{ij} = ∂y_i/∂x_j,  shape (B, n, n)
        J = vmap(jacrev(demand_fn, argnums=0))(
            x.detach().requires_grad_(True), c.detach()
        )

        jk = torch.triu_indices(n, n, offset=1, device=device)
        res = J - J.permute(0, 2, 1)             # (B, n, n)
        residuals_upper = res[:, jk[0], jk[1]]   # (B, n_pairs)

        per_sample = (residuals_upper ** 2).sum(dim=-1)
        penalty    = per_sample.mean()

        return CrossDerivativesOut(
            penalty=penalty,
            per_sample=per_sample,
            residuals_upper=residuals_upper,
        )