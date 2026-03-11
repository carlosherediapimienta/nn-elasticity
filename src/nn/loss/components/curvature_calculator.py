import torch
from torch import einsum

class CurvatureCalculator:
    """
    Computes d²y_i/dx_i² including own-price and cross-price interaction terms.

    Full curvature from the scalar potential:
      κ_i  =  Σ_k w_{ik}·B''_k(x_i)                                  [own]
            + Σ_{j: p=(i,j)} Σ_{k,l} u_{p,k,l}·B''_k(x_i)·B_l(x_j)  [i is left]
            + Σ_{j: p=(j,i)} Σ_{k,l} u_{p,k,l}·Ψ_k(x_j)·B''_l(x_i)  [i is right]

    Public API: run().
    """

    def run(
        self,
        w: torch.Tensor,      # (B, n, K)
        ddBx: torch.Tensor,   # (B, n, K)
        dddBx: torch.Tensor,  # (B, n, K)
        u: torch.Tensor | None,      # (B, n_cross, K, K) or empty
        Bx: torch.Tensor,     # (B, n, K)
        IBx: torch.Tensor,    # (B, n, K)
        pairs: torch.Tensor | None,  # (2, n_cross)
    ) -> torch.Tensor:        # (B, n)
        # We do curvature in float32 for numerical stability under AMP.
        w = w.float()
        ddBx = ddBx.float()
        B, n, _ = w.shape

        kappa = (w * ddBx).sum(-1)   # (B, n)

        # Fast path: no cross terms
        if (u is None) or (pairs is None) or (u.numel() == 0) or (pairs.numel() == 0):
            return kappa

        u = u.float()
        Bx = Bx.float()
        IBx = IBx.float()

        i_idx, j_idx = pairs[0], pairs[1]

        # i is "left" in pair (i<j): Σ_{k,l} u_{p,k,l} · B''_k(x_i) · B_l(x_j)
        left = einsum('bpk,bpkl,bpl->bp', ddBx[:, i_idx], u, Bx[:, j_idx])

        # i is "right" in pair (j<i): Σ_{k,l} u_{p,k,l} · Ψ_k(x_j) · B''_l(x_i)
        right = einsum('bpk,bpkl,bpl->bp', IBx[:, i_idx], u, dddBx[:, j_idx])

        i_exp = i_idx.unsqueeze(0).expand(B, -1)
        j_exp = j_idx.unsqueeze(0).expand(B, -1)
        kappa = kappa.scatter_add(1, i_exp, left.to(kappa.dtype))
        kappa = kappa.scatter_add(1, j_exp, right.to(kappa.dtype))

        return kappa  # (B, n)