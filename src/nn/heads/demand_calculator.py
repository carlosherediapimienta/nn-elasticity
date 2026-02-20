import torch

class DemandCalculator:
    """
    Calculate predicted demand (log-space) using the extended potential model:

      y_hat_i = b_i(c) + beta_i(c)*x_i + Σ_k w_{ik}(c)*B_k(x_i) + Σ_{j≠i} A_{ij}(c)*x_j

    Derives from the scalar potential:
      Φ(x,c) = Σ_i [b_i*x_i + beta_i/2*x_i² + Σ_k w_{ik}*∫B_k dx_i] + (1/2)*x^T A x

    so y = ∂Φ/∂x, guaranteeing ∂y_i/∂x_j = ∂y_j/∂x_i by Schwarz's theorem.

    Public API: run().
    """

    def run(
        self,
        b: torch.Tensor,       # (B, n)
        beta: torch.Tensor,    # (B, n)
        w: torch.Tensor,       # (B, n, K)
        x: torch.Tensor,       # (B, n)
        Bx: torch.Tensor,      # (B, n, K)
        dBx: torch.Tensor,     # (B, n, K)
        ddBx: torch.Tensor,    # (B, n, K)
        IBx: torch.Tensor,     # (B, n, K)
        u: torch.Tensor,       # (B, n_cross, K, K)
        pairs: torch.Tensor,   # (2, n_cross)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            y_hat:   (B, n)    predicted demand
            eps_hat: (B, n)    own-price elasticity (diagonal of E)
            E:       (B, n, n) full elasticity matrix ∂y_i/∂x_j
        """
        B, n, K = Bx.shape
        i_idx, j_idx = pairs[0], pairs[1]   # (n_cross,)

        # ── Own-price terms ──────────────────────────────────────────────────
        y_hat   = b + beta * x + (w * Bx).sum(dim=-1)   # (B, n)
        eps_hat = beta + (w * dBx).sum(dim=-1)           # (B, n)

        # ── Cross-product spline terms ────────────────────────────────────────
        Bx_i   = Bx[:,  i_idx, :]   # (B, n_cross, K)
        Bx_j   = Bx[:,  j_idx, :]
        dBx_i  = dBx[:, i_idx, :]
        dBx_j  = dBx[:, j_idx, :]
        ddBx_j = ddBx[:,j_idx, :]
        IBx_i  = IBx[:, i_idx, :]

        # y_i  += Σ_{k,l} u_{p,k,l} * B_k(x_i)  * B_l(x_j)
        contrib_yi = torch.einsum('bpk,bpkl,bpl->bp', Bx_i,  u, Bx_j).to(Bx.dtype)

        # y_j  += Σ_{k,l} u_{p,k,l} * Ψ_k(x_i)  * B'_l(x_j)
        contrib_yj = torch.einsum('bpk,bpkl,bpl->bp', IBx_i, u, dBx_j).to(Bx.dtype)

        # eps_i += Σ_{k,l} u_{p,k,l} * B'_k(x_i) * B_l(x_j)
        contrib_ei = torch.einsum('bpk,bpkl,bpl->bp', dBx_i, u, Bx_j).to(Bx.dtype)

        # eps_j += Σ_{k,l} u_{p,k,l} * Ψ_k(x_i)  * B''_l(x_j)
        contrib_ej = torch.einsum('bpk,bpkl,bpl->bp', IBx_i, u, ddBx_j).to(Bx.dtype)

        i_exp = i_idx.unsqueeze(0).expand(B, -1)
        j_exp = j_idx.unsqueeze(0).expand(B, -1)

        y_hat   = y_hat.scatter_add(1, i_exp, contrib_yi)
        y_hat   = y_hat.scatter_add(1, j_exp, contrib_yj)
        eps_hat = eps_hat.scatter_add(1, i_exp, contrib_ei)
        eps_hat = eps_hat.scatter_add(1, j_exp, contrib_ej)

        # ── Elasticity matrix E[b, i, j] = ∂y_i/∂x_j ────────────────────────
        E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)

        # Diagonal: own-price
        E[:, torch.arange(n), torch.arange(n)] = eps_hat

        # Off-diagonal: cross-price  E_{ij} = Σ_{k,l} u_{p,k,l} * B_k(x_i) * B'_l(x_j)
        E_cross = torch.einsum('bpk,bpkl,bpl->bp', Bx_i, u, dBx_j)   # (B, n_cross)
        E[:, i_idx, j_idx] = E_cross.to(E.dtype)   # cast float16 → float32
        E[:, j_idx, i_idx] = E_cross.to(E.dtype)

        return y_hat, eps_hat, E