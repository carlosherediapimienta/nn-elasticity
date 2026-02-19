import torch
import torch.nn as nn
from typing import Optional, Dict

from ..diagnostics.closure_diagnostics import ClosureDiagnostics


class ElasticityFirstLoss(nn.Module):
    """
    Loss para baseline elasticity-first (Sección 3.4):

      L = Huber(y_hat - y_true)
        + lambda_cl * R_cl(E_phi)
        + lambda_pos * mean_i ReLU(E_ii)

    donde R_cl penaliza violaciones de cierre:
      c^i_{jk} = ∂_k E_{i,j} - ∂_j E_{i,k}, j<k.
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        lambda_cl: float = 0.0,
        lambda_pos: float = 0.0,
        closure_pair_subsample: Optional[int] = None,
    ):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="none")
        self.lambda_cl = float(lambda_cl)
        self.lambda_pos = float(lambda_pos)
        self.closure = ClosureDiagnostics(pair_subsample=closure_pair_subsample)

    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        x: torch.Tensor,
        E_model: nn.Module,
        c: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_hat: (B, n)
            y_true: (B, n)
            x: (B, n) punto(s) donde evaluamos regularizadores (normalmente el endpoint observado)
            E_model: módulo que devuelve E(x,c): (B,n,n)
            c: (B, d) o None

        Returns:
            dict con {'loss', 'fit', 'closure', 'pos'} (todos escalares)
        """
        if y_hat.shape != y_true.shape:
            raise ValueError(f"y_hat y y_true deben tener el mismo shape, got {y_hat.shape} vs {y_true.shape}")

        # Fit: Huber en log-space, sum en dimensión de demanda, mean en batch
        fit_per_elem = self.huber(y_hat, y_true)           # (B, n)
        fit = fit_per_elem.sum(dim=-1).mean()

        # Positivity penalty: ReLU de diagonal (own elasticities)
        E = E_model(x, c)                                  # (B, n, n)
        diag = torch.diagonal(E, dim1=-2, dim2=-1)          # (B, n)
        pos = torch.relu(diag).mean()

        # Closure penalty (opcional)
        if self.lambda_cl != 0.0:
            cl_out = self.closure.run(E_model=E_model, x=x, c=c)
            closure = cl_out.penalty
        else:
            closure = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = fit + self.lambda_pos * pos + self.lambda_cl * closure

        return {
            "loss": loss,
            "fit": fit.detach(),
            "pos": pos.detach(),
            "closure": closure.detach(),
        }
