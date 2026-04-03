import torch

class DemandCalculator:
    """
    Computes predicted demand (log-space) from the parameters b, beta, w, u.

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
            E:       (B, n, n) full elasticity matrix \partial y_i / \partial x_j (or None if return_E=False)
        """
        B, n, K = Bx.shape # Number of samples in the batch, number of products, number of splines.

        # ── Own-price terms ──────────────────────────────────────────────────
        y_hat   = b + beta * x + (w * Bx).sum(dim=-1)   # (B, n)
        eps_hat = beta + (w * dBx).sum(dim=-1)          # (B, n)
        # Recall that dim=-1 means the last dimension,
        # namely, the sum is over the splines.

        # Fast path: no cross terms
        # We check if there are cross-price terms.
        has_cross = (u is not None) and (pairs is not None) and (pairs.numel() > 0) and (u.numel() > 0)
        # If there are no cross-price terms, we return the own-price terms.
        # The elasticity matrix E is only computed the diagonal, otherwise, 0.
        if not has_cross:
            if return_E:
                E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype) # (B, n, n)
                # E[:, 0, 0] = eps_hat[:, 0]  -  own-price elasticity of product 0
                # E[:, 1, 1] = eps_hat[:, 1]  - own-price elasticity of product 1
                # ...
                # E[:, n-1, n-1] = eps_hat[:, n-1] - own-price elasticity of product n-1
                # Namely,
                # E[b] = [[eps_0,   0,     0  ]
                #     [  0,   eps_1,   0  ]
                #     [  0,     0,   eps_2]]
                # This is equivalent to:
                E[:, torch.arange(n), torch.arange(n)] = eps_hat
            else:
                E = None
            return y_hat, eps_hat, E

        # If there are cross-price terms, we compute the cross-price terms.
        i_idx, j_idx = pairs[0], pairs[1]   # (n_cross,)

        # ── Cross-product spline terms ────────────────────────────────────────
        # We get the spline bases and derivatives for the cross-price terms (i,j).
        Bx_i   = Bx[:,  i_idx, :]   # (B, n_cross, K)
        Bx_j   = Bx[:,  j_idx, :]
        dBx_i  = dBx[:, i_idx, :]
        dBx_j  = dBx[:, j_idx, :]
        ddBx_j = ddBx[:,j_idx, :]
        IBx_i  = IBx[:, i_idx, :]

        # -- THEORY IMPLEMENTATION: See Article ────────────────────────────────────

        # y_i  += Σ_{k,l} u_{p,k,l} * B_k(x_i)  * B_l(x_j)
        contrib_yi = torch.einsum('bpk,bpkl,bpl->bp', Bx_i,  u, Bx_j).to(Bx.dtype)

        # y_j  += Σ_{k,l} u_{p,k,l} * Ψ_k(x_i)  * B'_l(x_j)
        contrib_yj = torch.einsum('bpk,bpkl,bpl->bp', IBx_i, u, dBx_j).to(Bx.dtype)

        # eps_i += Σ_{k,l} u_{p,k,l} * B'_k(x_i) * B_l(x_j)
        contrib_ei = torch.einsum('bpk,bpkl,bpl->bp', dBx_i, u, Bx_j).to(Bx.dtype)

        # eps_j += Σ_{k,l} u_{p,k,l} * Ψ_k(x_i)  * B''_l(x_j)
        contrib_ej = torch.einsum('bpk,bpkl,bpl->bp', IBx_i, u, ddBx_j).to(Bx.dtype)

        # Recall that: .einsum() is a way to perform a sum of products of tensors.
        # For instance, if we have:
        # a = torch.tensor([[1, 2], [3, 4]])
        # b = torch.tensor([[5, 6], [7, 8]])
        # c = torch.tensor([[9, 10], [11, 12]])
        # Then, torch.einsum('ij,ij->ij', a, b) is equivalent to:
        # [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
        # Namely, it is a sum of products of the elements of the tensors.
        # For our case, we have the sum, for intance, 'bpk,bpkl,bpl->bp' 
        # keeping the dimension (b,p) but summing over the other dimensions.

        # Now, y_hat and eps_hat are tensor of shape (B, n), but contrib_ei,
        # contrib_ej, contrib_yi and contrib_yj are tensor of shape (B, n_cross), namely,
        # a contribution for each cross-price term. Therefore, we need to "scatter" 
        # the contributions into the product i or j.

        # We start from: (n_cross,) -> (1, n_cross) -> (B, n_cross)
        # For instance, 
        # i_idx = [0, 0, 1]   # form (3,) = (n_cross,)
        # j_idx = [1, 2, 2]   # form (3,) = (n_cross,)
        # Then, .unsqueeze(0):
        # i_idx.unsqueeze(0) = [[0, 0, 1]]
        # j_idx.unsqueeze(0) = [[1, 2, 2]]
        # Then, .expand(B, -1):
        # i_idx.unsqueeze(0).expand(B, -1) = [[0, 0, 1], [0, 0, 1]]
        # j_idx.unsqueeze(0).expand(B, -1) = [[1, 2, 2], [1, 2, 2]]
        # because B = 2.
        i_exp = i_idx.unsqueeze(0).expand(B, -1) # (B, n_cross)
        j_exp = j_idx.unsqueeze(0).expand(B, -1) # (B, n_cross)

        # Now, for instance:
        # contrib_yi = [[0.1, 0.2, 0.3],  
        #              [0.4, 0.5, 0.6]] 
        # and 
        # i_exp = [[0, 0, 1], [0, 0, 1]] # pair0->prod0, pair1->prod0, pair2->prod1
        # and 
        # y_hat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        # For each sample in the batch,
        # we add the contribution to the product. Namely,
        # prod 0 += contrib_yi[0, 0] = 0.1   (pair 0 → prod 0)
        # prod 0 += contrib_yi[0, 1] = 0.2   (pair 1 → prod 0)
        # prod 1 += contrib_yi[0, 2] = 0.3   (pair 2 → prod 1)
        # prod 2 += nothing              = 0.0 (no contribution)
        # For the second sample in the batch,
        # prod 0 += 0.4   (pair 0 → prod 0)
        # prod 0 += 0.5   (pair 1 → prod 0)
        # prod 1 += 0.6   (pair 2 → prod 1)
        # prod 2 += nothing              = 0.0 (no contribution)
        # Final result:
        # y_hat = [[1.0 + 0.1 + 0.2,  2.0 + 0.3,  3.0],   # sample 0
        #          [4.0 + 0.4 + 0.5,  5.0 + 0.6,  6.0]]    # sample 1
        #        = [[1.3,  2.3,  3.0], [4.9,  5.6,  6.0]]
        y_hat   = y_hat.scatter_add(1, i_exp, contrib_yi)
        y_hat   = y_hat.scatter_add(1, j_exp, contrib_yj)
        eps_hat = eps_hat.scatter_add(1, i_exp, contrib_ei)
        eps_hat = eps_hat.scatter_add(1, j_exp, contrib_ej)

        # ── Elasticity matrix E[b, i, j] = \partial y_i / \partial x_j ────────────────────────
        if return_E:
            E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)

            # Diagonal: own-price
            E[:, torch.arange(n), torch.arange(n)] = eps_hat

            # Off-diagonal: cross-price  E_{ij} = Σ_{k,l} u_{p,k,l} * B_k(x_i) * B'_l(x_j)
            E_cross = torch.einsum('bpk,bpkl,bpl->bp', Bx_i, u, dBx_j)   # (B, n_cross)
            # Symmetry E_{ij} = E_{ji} holds exactly by construction,
            # so both triangles of E share the same values.
            E[:, i_idx, j_idx] = E_cross.to(E.dtype)
            E[:, j_idx, i_idx] = E_cross.to(E.dtype)
        else:
            E = None

        return y_hat, eps_hat, E