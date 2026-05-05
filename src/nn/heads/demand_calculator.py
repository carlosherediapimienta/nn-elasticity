import torch

class DemandCalculator:
    """
    Computes predicted demand (log-space) and elasticities from
    the parameters b, beta, w, beta_cross, w_cross, u.

    Public API: run().
    """
    def run(
        self,
        b: torch.Tensor,           # (B, n)
        beta: torch.Tensor,        # (B, n)
        w: torch.Tensor,           # (B, n, K)
        x: torch.Tensor,           # (B, n)
        Bx: torch.Tensor,          # (B, n, K)
        dBx: torch.Tensor,         # (B, n, K)
        beta_cross: torch.Tensor,  # (B, n_cross) - linear cross-price coefficient β_{ij}(x)
        w_cross: torch.Tensor,     # (B, n_cross, K) - cross-price spline weights w_{ij}(x)
        u: torch.Tensor,           # (B, n_cross, K, K)
        pairs: torch.Tensor,       # (2, n_cross)
        attn_weights: torch.Tensor | None = None,  # (B, n_cross)
        return_E: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            y_hat:   (B, n)    predicted demand
            eps_hat: (B, n)    own-price elasticity (diagonal of E)
            E:       (B, n, n) full elasticity matrix \partial g_i / \partial u_j
                               (or None if return_E=False)
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
        x_i    = x[:, i_idx]
        x_j    = x[:, j_idx]

        # -- THEORY IMPLEMENTATION: See Article ────────────────────────────────────

        # g_i += a_ij · [ β_{ij} · u_j  +  w_{ij}^T B_j(u_j)  +  B_i(u_i)^T U^{(ij)} B_j(u_j) ]
        # (attention a_ij is applied below after this block)
        contrib_yi = (
            beta_cross * x_j                                          # linear cross term
            + (w_cross * Bx_j).sum(dim=-1)                           # spline cross term
            + torch.einsum('bpk,bpkl,bpl->bp', Bx_i, u, Bx_j)      # bilinear term
        ).to(Bx.dtype)

        # E_{ii} += a_ij · B'_i(u_i)^T U^{(ij)} B_j(u_j)
        # Note: β_{ij} · u_j and w_{ij}^T B_j(u_j) do not depend on u_i,
        # so they do not contribute to the own-price elasticity.
        contrib_ei = (
            torch.einsum('bpk,bpkl,bpl->bp', dBx_i, u, Bx_j)
        ).to(Bx.dtype)

        # If attn_weights is not None, we multiply the contributions by the attention weights.
        if attn_weights is not None:
            contrib_yi = contrib_yi * attn_weights.to(Bx.dtype)
            contrib_ei = contrib_ei * attn_weights.to(Bx.dtype)

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
        eps_hat = eps_hat.scatter_add(1, i_exp, contrib_ei)

        # ── Elasticity matrix E[b, i, j] = \partial y_i / \partial x_j ────────────────────────
        if return_E:
            E = torch.zeros(B, n, n, device=Bx.device, dtype=Bx.dtype)

            # Diagonal: own-price
            E[:, torch.arange(n), torch.arange(n)] = eps_hat

            # Off-diagonal: cross-price
            # E_{ij} = a_ij · [ β_{ij}  +  w_{ij}^T B'_j(u_j)  +  B_i(u_i)^T U^{(ij)} B'_j(u_j) ]
            E_cross = (
                beta_cross                                                # linear term
                + (w_cross * dBx_j).sum(dim=-1)                          # spline cross term
                + torch.einsum('bpk,bpkl,bpl->bp', Bx_i, u, dBx_j)     # bilinear term
            ).to(E.dtype)  # (B, n_cross)

            # If attn_weights is not None, we multiply the contributions by the attention weights.
            if attn_weights is not None:
                E_cross = E_cross * attn_weights.to(E.dtype)

            # With directed pairs (i, j), off-diagonal elasticities are directional:
            # E_{ij} and E_{ji} are learned independently (no enforced symmetry).
            E[:, i_idx, j_idx] = E_cross
        else:
            E = None

        return y_hat, eps_hat, E