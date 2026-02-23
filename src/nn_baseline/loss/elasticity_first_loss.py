import torch
import torch.nn as nn
from typing import Optional, Dict

from ..diagnostics.closure_diagnostics import ClosureDiagnostics
from ..utils import eval_dropouts


class ElasticityFirstLoss(nn.Module):
    """
    Loss for baseline elasticity-first (Section 3.4):

      L = Huber(y_hat - y_true)
        + lambda_cl * R_cl(E_phi)
        + lambda_pos * mean_i ReLU(E_ii)

    where R_cl penalizes violations of closure:
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
            x: (B, n) point(s) where we evaluate regularizers (normally the observed endpoint)
            E_model: module that returns E(x,c): (B,n,n)
            c: (B, d) or None

        Returns:
            dict with {'loss', 'fit', 'closure', 'pos'} (all scalars)
        """
        if y_hat.shape != y_true.shape:
            raise ValueError(f"y_hat and y_true must have the same shape, got {y_hat.shape} vs {y_true.shape}")

        # Fit: Huber in log-space, sum in demand dimension, mean in batch
        fit_per_elem = self.huber(y_hat, y_true)           # (B, n)
        fit = fit_per_elem.sum(dim=-1).mean()

        # Positivity penalty: ReLU de diagonal (own elasticities)
        with eval_dropouts(E_model):
            E = E_model(x, c)                              # (B, n, n)
        diag = torch.diagonal(E, dim1=-2, dim2=-1)
        pos = torch.relu(diag).mean()

        # Closure penalty (optional)
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
