import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import FitLoss, SmoothnessPenalty

class ElasticityLoss(nn.Module):
    """
    Compound loss for the elasticity model:
      loss = L_fit + λ_smooth · L_smooth + λ_elast · L_elast
      where:
        - L_fit:    Huber fit loss on observed log-demands.
        - L_smooth: smoothness penalty on the curvature of own-price demand curves.
        - L_elast:  elasticity bound penalty — soft constraints on own-price and
                    cross-price elasticities via asymmetric squared hinge losses:
                        [l_own,  r_own]  = [-5, 0]  for E_{ii}
                        [l_cross, r_cross] = [-1, 1]  for E_{ij}, i≠j

    Public API: run().
    """
    def __init__(
        self,
        huber_delta: float = 1.0,
        lambda_smooth: float = 0.0,
        lambda_elast: float = 0.0,
        l_own: float = -5.0,       # lower bound for own-price elasticities
        r_own: float = 0.0,        # upper bound for own-price elasticities
        l_cross: float = -1.0,     # lower bound for cross-price elasticities
        r_cross: float = 1.0,      # upper bound for cross-price elasticities
        rho_own_low: float = 1.0,  # asymmetric weight for own-price lower-bound violations
                                   # set > 1 to penalise E_{ii} << l_own more strongly
        reduction: str = "mean",
    ):
        super().__init__()
        self.fit_loss           = FitLoss(delta=huber_delta, reduction=reduction)
        self.smoothness_penalty = SmoothnessPenalty()
        self.lambda_smooth      = float(lambda_smooth)
        self.lambda_elast       = float(lambda_elast)
        self.l_own,   self.r_own   = float(l_own),   float(r_own)
        self.l_cross, self.r_cross = float(l_cross), float(r_cross)
        self.rho_own_low           = float(rho_own_low)

    def run(
        self,
        y_hat: torch.Tensor,     # (B, n)
        y_true: torch.Tensor,    # (B, n)
        obs_mask: torch.Tensor,  # (B, n)
        w: torch.Tensor,         # (B, n, K)
        ddBx: torch.Tensor,      # (B, n, K)
        u: torch.Tensor,         # (B, n_cross, K, K)
        Bx: torch.Tensor,        # (B, n, K)
        pairs: torch.Tensor,     # (2, n_cross)
        E: torch.Tensor | None = None,  # (B, n, n) full elasticity matrix — required if lambda_elast > 0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        # 1. Fit loss — Huber on observed log-demands only.
        mask = obs_mask.bool()
        if mask.any():
            loss_fit = self.fit_loss.run(y_hat[mask], y_true[mask])
        else:
            loss_fit = y_hat.new_tensor(0.0)

        # 2. Smoothness penalty — penalises high curvature of own-price demand curves.
        if self.lambda_smooth > 0.0:
            loss_smooth = self.smoothness_penalty.run(w, ddBx, u, Bx, pairs)
        else:
            loss_smooth = y_hat.new_tensor(0.0)

        # 3. Elasticity bound penalty — soft [l, r] constraints on E_{ij}.
        #
        # For each (i, j) entry we apply an asymmetric squared hinge:
        #   ReLU(E_{ij} - r_{ij})^2   — penalises values above the upper bound
        #   ρ_{ij} · ReLU(l_{ij} - E_{ij})^2  — penalises values below the lower bound
        #
        # Bounds:
        #   own-price  (i=j): [l_own,  r_own]  = [-5,  0]
        #   cross-price (i≠j): [l_cross, r_cross] = [-1,  1]
        #
        # Mask M_{ij}:
        #   own-price:   m_i             (product i observed)
        #   cross-price: m_i · m_j · 1[(i,j) active in the sparse graph]
        #   N_E = sum of all active mask entries (normalisation denominator)
        diag = torch.arange(E.shape[1], device=E.device) if E is not None else None

        if self.lambda_elast > 0.0 and E is not None:
            B, n, _ = E.shape

            # Build per-entry bounds and asymmetry weights — shape (n, n).
            L   = E.new_full((n, n), self.l_cross)
            R   = E.new_full((n, n), self.r_cross)
            rho = E.new_ones(n, n)
            L[diag, diag]   = self.l_own
            R[diag, diag]   = self.r_own
            rho[diag, diag] = self.rho_own_low

            # Build mask M_{ij} — shape (B, n, n).
            # Diagonal (own-price): M_{ii} = m_i.
            # Off-diagonal (cross-price): M_{ij} = m_i · m_j · 1[(i,j) in graph].
            # The sparse graph membership is implicitly encoded in E: positions not
            # selected by SparseNeighborSelector are left as zero by DemandCalculator,
            # so their penalty is zero regardless of the mask value.
            m = obs_mask.float()                          # (B, n)
            M = m.unsqueeze(2) * m.unsqueeze(1)           # (B, n, n): m_i · m_j

            # Compute the per-entry hinge penalties.
            upper_viol = F.relu(E - R.unsqueeze(0)) ** 2                   # (B, n, n)
            lower_viol = F.relu(L.unsqueeze(0) - E) ** 2                   # (B, n, n)
            penalty    = M * (upper_viol + rho.unsqueeze(0) * lower_viol)  # (B, n, n)

            N_E        = M.sum().clamp(min=1.0)  # avoid division by zero
            loss_elast = penalty.sum() / N_E
        else:
            loss_elast = y_hat.new_tensor(0.0)

        loss = (loss_fit
                + self.lambda_smooth * loss_smooth
                + self.lambda_elast  * loss_elast)

        # Monitoring stats — detached to free the computation graph.
        # eps_hat is the diagonal of E; fall back to zeros if E is not available.
        eps_hat = E[:, diag, diag].detach() if E is not None else y_hat.new_zeros(y_hat.shape)
        logs = {
            "loss":        loss.detach(),
            "loss_fit":    loss_fit.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_elast":  loss_elast.detach(),
            "eps_mean":    eps_hat.mean(),
            "eps_p50":     eps_hat.median(),
            "obs_frac":    obs_mask.mean().detach(),
        }
        return loss, logs
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)