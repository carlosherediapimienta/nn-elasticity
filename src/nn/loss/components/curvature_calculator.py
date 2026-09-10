import torch
from torch import einsum

class CurvatureCalculator:
    """
    Computes d^2log(y_i)/d(log(x_i))^2 including own-price and cross-price interaction terms.
    Public API: run().
    """
    def run(
        self,
        w: torch.Tensor,      # (B, n, K)
        ddBx: torch.Tensor,   # (B, n, K)
        u: torch.Tensor | None,      # (B, n_cross, K, K) or empty
        Bx: torch.Tensor,     # (B, n, K)
        pairs: torch.Tensor | None,  # (2, n_cross)
        attn_weights: torch.Tensor | None = None, # (B, n_cross)
        availability: torch.Tensor | None = None, # (B, n) bool
    ) -> torch.Tensor:        # (B, n)

        # We do curvature in float32 for numerical stability under AMP.
        w = w.float()
        ddBx = ddBx.float()
        B, n, _ = w.shape

        # Important! ------------------------------------------------------------
        # The curvature is analytically computed in the article, therefore, 
        # kindly refer to the article for more details.
        # ----------------------------------------------------------------------

        # Compute the first part of the curvature of the demand curve.
        kappa = (w * ddBx).sum(-1)   # (B, n)

        # Fast path: no cross terms
        # If there are no cross-price terms, we return the own-price terms.
        if (u is None) or (pairs is None) or (u.numel() == 0) or (pairs.numel() == 0):
            return kappa

        # Convert the tensors to float32 for numerical stability under AMP.
        u = u.float()
        Bx = Bx.float()

        # Get the indices of the cross-price terms.
        i_idx, j_idx = pairs[0], pairs[1]

        # Compute the contribution of the cross-price terms to the curvature.
        # For more details how it works these two lines, please, refer to the DemandCalculator class.
        
        # Curvature of the demand curve.
        # κ_i += B''(x_i)^T U^{(ij)} B(x_j)
        contrib = torch.einsum('bpk,bpkl,bpl->bp', ddBx[:, i_idx], u, Bx[:, j_idx])
        if attn_weights is not None:
            # k_i = a_ij * k_i
            contrib = contrib * attn_weights.float()

        if availability is not None:
            avail_j = availability[:, j_idx].to(contrib.dtype)
            contrib = contrib * avail_j

        # Add the contributions to the curvature.
        # This step is exactly the same as the one in the DemandCalculator class.
        # So, please, refer to the DemandCalculator class for more details.
        i_exp   = i_idx.unsqueeze(0).expand(B, -1)
        kappa   = kappa.scatter_add(1, i_exp, contrib.to(kappa.dtype))

        return kappa  # (B, n)