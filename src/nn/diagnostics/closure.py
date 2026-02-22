import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.func import vmap, jacrev


@dataclass
class ClosureOut:
    """Result of the closure diagnostic."""
    penalty: torch.Tensor          # scalar: mean of ||c^2||
    per_sample: torch.Tensor       # (B,)
    residuals_upper: torch.Tensor  # (B, n, n_pairs)


class ClosureDiagnostics:
    """
    Calculates closure residuals: c^i_{jk} = ∂_k E_{i,j} - ∂_j E_{i,k}, for j<k.
    Penalizes sum_{i,j<k} c^2. Cost ~ O(B * n^3).
    Requires PyTorch >= 2.0 (torch.func).
    Public API: run()
    """

    def run(self, model: nn.Module, batch: dict) -> ClosureOut:
        """
        Args:
            model: ICDN instance
            batch: dict with all batch tensors

        Returns:
            ClosureOut(penalty, per_sample, residuals_upper)
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

        # D[b, i, j, k] = ∂_{x_k} ∂_{x_j} y_i
        D = vmap(jacrev(jacrev(demand_fn, argnums=0), argnums=0))(
            x.detach().requires_grad_(True), c.detach()
        )  # (B, n, n, n)

        # Schwarz residual: D[b,i,j,k] - D[b,i,k,j]  →  should be ~0
        res = D - D.permute(0, 1, 3, 2)  # (B, n, n, n)

        jk = torch.triu_indices(n, n, offset=1, device=device)
        residuals_upper = res[:, :, jk[0], jk[1]]           # (B, n, n_pairs)
        per_sample = (residuals_upper ** 2).mean(dim=(1, 2))
        penalty = per_sample.mean()
        
        return ClosureOut(
            penalty=penalty,
            per_sample=per_sample,
            residuals_upper=residuals_upper,
        )