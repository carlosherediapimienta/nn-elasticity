import torch

class DemandCalculator:
    """
    Computes predicted demand (log-space) from the scalar potential:

      Φ(x,c) = Σ_i [b_i·x_i + beta_i/2·x_i² + Σ_k w_{ik}·Ψ_k(x_i)]
             + Σ_{p=(i<j)} Σ_{k,l} u_{p,k,l} · Ψ_k(x_i) · B_l(x_j)

    where Ψ_k(x) = ∫B_k(x)dx  (antiderivative of the spline basis, = IBx).

    Demand is y = ∂Φ/∂x, which guarantees ∂y_i/∂x_j = ∂y_j/∂x_i
    (Slutsky symmetry) by Schwarz's theorem, with no explicit constraint on u.

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
        return_E: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            y_hat:   (B, n)    predicted demand
            eps_hat: (B, n)    own-price elasticity (diagonal of E)
            E:       (B, n, n) full elasticity matrix ∂y_i/∂x_j (or None if return_E=False)
        """
        B, n, K = Bx.shape

        # ── Own-price terms ──────────────────────────────────────────────────
        y_hat   = b + beta * x + (w * Bx).sum(dim=-1)   # (B, n)
        eps_hat = beta + (w * dBx).sum(dim=-1)          # (B, n)

        # Fast path: no cross terms
        has_cross = (u is not None) and (pairs is not None) and (pairs.numel() > 0) and (u.numel() > 0)
        if not has_cross:
            if return_E:
                E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)
                E[:, torch.arange(n), torch.arange(n)] = eps_hat
            else:
                E = None
            return y_hat, eps_hat, E

        i_idx, j_idx = pairs[0], pairs[1]   # (n_cross,)

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
        if return_E:
            E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)

            # Diagonal: own-price
            E[:, torch.arange(n), torch.arange(n)] = eps_hat

            # Off-diagonal: cross-price  E_{ij} = Σ_{k,l} u_{p,k,l} * B_k(x_i) * B'_l(x_j)
            E_cross = torch.einsum('bpk,bpkl,bpl->bp', Bx_i, u, dBx_j)   # (B, n_cross)
            E[:, i_idx, j_idx] = E_cross.to(E.dtype)
            E[:, j_idx, i_idx] = E_cross.to(E.dtype)
        else:
            E = None

        return y_hat, eps_hat, E