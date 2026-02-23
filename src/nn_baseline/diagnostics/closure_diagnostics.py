import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from torch.func import vmap, jacrev
from ..utils import eval_dropouts


@dataclass
class ClosureOut:
    """Result of the closure diagnostic (only data)."""
    penalty: torch.Tensor           # escalar
    per_sample: torch.Tensor        # (B,)
    residuals_upper: torch.Tensor   # (B, n, n_pairs)


class ClosureDiagnostics:
    """
    Calculates closure residuals: c^i_{jk} = ∂_k E_{i,j} - ∂_j E_{i,k}, for j<k.
    Penalizes sum_{i,j<k} c^2. Cost ~ O(B * n^3).
    Requires PyTorch >= 2.0 (torch.func).
    Public API: run()
    """

    def __init__(self, pair_subsample: Optional[int] = None):
        """
        Args:
            pair_subsample: if specified, sample randomly this number
                            of pairs (j,k) instead of using all (useful for n large)
        """
        self.pair_subsample = pair_subsample

    def run(
        self,
        E_model: nn.Module,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> ClosureOut:
        """
        Args:
            E_model: module that returns E(x, c): (B, n, n)
            x: (B, n) log-prices (no requires requires_grad)
            c: (B, d) vector of context, or None

        Returns:
            ClosureOut with penalty, per_sample and residuals_upper
        """
        B, n = x.shape
        device = x.device

        jk = torch.triu_indices(n, n, offset=1, device=device)
        if self.pair_subsample is not None and self.pair_subsample < jk.shape[1]:
            idx = torch.randperm(jk.shape[1], device=device)[:self.pair_subsample]
            jk = jk[:, idx]

        def E_single(x1: torch.Tensor, c1: Optional[torch.Tensor]) -> torch.Tensor:
            xs = x1.unsqueeze(0)
            cs = None if c1 is None else c1.unsqueeze(0)
            return E_model(xs, cs)[0]

        with eval_dropouts(E_model):
            if c is None:
                dE_dx = vmap(jacrev(lambda x1: E_single(x1, None)))(
                    x.detach().requires_grad_(True)
                )
            else:
                dE_dx = vmap(jacrev(lambda x1, c1: E_single(x1, c1), argnums=0))(
                    x.detach().requires_grad_(True), c.detach()
                )
            
        res = dE_dx - dE_dx.permute(0, 1, 3, 2)
        residuals_upper = res[:, :, jk[0], jk[1]]
        per_sample = (residuals_upper ** 2).sum(dim=(1, 2))
        penalty = per_sample.mean()

        return ClosureOut(
            penalty=penalty,
            per_sample=per_sample,
            residuals_upper=residuals_upper,
        )