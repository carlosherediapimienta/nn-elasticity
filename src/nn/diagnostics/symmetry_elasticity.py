import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.func import vmap, jacrev


@dataclass
class SymmetryElasticityOut:
    """Result of the cross-derivatives symmetry diagnostic."""
    penalty: torch.Tensor          # scalar: mean of ||J - J^T||²
    per_sample: torch.Tensor       # (B,)
    residuals_upper: torch.Tensor  # (B, n_pairs): upper triangle of J - J^T


class SymmetryElasticityDiagnostics:
    """
    Certifies Slutsky symmetry of the demand Jacobian:
      S_{ij} = ∂y_i/∂x_j - ∂y_j/∂x_i  →  debe ser 0.

    Para el ICDN extendido con matriz A simétrica:
      J_{ij} = A_{ij}(c) = A_{ji}(c)  →  J = J^T exactamente por construcción.

    Penalty = mean_B( Σ_{i<j} S_{ij}² )  →  debe ser 0.0 para ICDN.

    Public API: run()
    """

    def run(self, model: nn.Module, batch: dict) -> SymmetryElasticityOut:
        model.eval()
        device = next(model.parameters()).device
        n = model.n

        with torch.no_grad():
            _, _, aux = model.run(batch, return_parts=True)
            E = aux['E']   # (B, n, n) — ya calculado analíticamente

        jk = torch.triu_indices(n, n, offset=1, device=device)
        res = E - E.permute(0, 2, 1)              # (B, n, n)  — debería ser ~0
        residuals_upper = res[:, jk[0], jk[1]]   # (B, n_pairs)

        per_sample = (residuals_upper ** 2).sum(dim=-1)
        penalty    = per_sample.mean()

        return SymmetryElasticityOut(
            penalty=penalty,
            per_sample=per_sample,
            residuals_upper=residuals_upper,
        )
        