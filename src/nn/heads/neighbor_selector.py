import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseNeighborSelector(nn.Module):
    """
    Sparse neighbor graph (batch-shared) + per-sample attention weights.

    Two-stage design:
      1. Structural candidates: batch-shared pairs selected by aggregated
         scores + metadata bonus (same category required, brand/style/liters bonus).
      2. Per-sample soft weights: softmax of q(h_i)^T k(h_j) within each
         focal product i's candidates, giving heterogeneous edge strengths
         per observation without breaking vectorization.

    Args:
        d_hidden:                   dimension of latent h from SharedProductEncoder
        d_attn:                     dimension of query/key projections
        k_neighbors:                number of neighbors to select per focal product i
        init_brand_bonus:           initial additive bonus for same brand_family_norm
        init_style_bonus:           initial additive bonus for same style_segment_norm
        init_liters_bonus:          initial additive bonus for similar liters_per_upc
        liters_gamma:               decay rate for liters similarity (exp(-gamma * log_dist))
        use_same_category_strict:   if True, same-category candidates are preferred first

    Returns (from run):
        pairs:        (2, E)   directed edges, E = n * k_eff, batch-shared
        edge_weights: (B, E)   per-sample soft weight per edge

    Public API: run()
    """

    def __init__(
        self,
        d_hidden: int,
        d_attn: int = 16,
        k_neighbors: int = 2,
        init_brand_bonus: float = 0.30,
        init_style_bonus: float = 0.10,
        init_liters_bonus: float = 0.10,
        liters_gamma: float = 1.0,
        use_same_category_strict: bool = True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_hidden, d_attn, bias=False)
        self.k_proj = nn.Linear(d_hidden, d_attn, bias=False)
        self.scale  = math.sqrt(d_attn)

        self.k                        = k_neighbors
        self.gamma                    = liters_gamma
        self.use_same_category_strict = use_same_category_strict

        # Raw parameters; effective bonus = softplus(raw) > 0 always.
        # Initialized via inv_softplus to recover the intended initial values.
        self.brand_bonus_raw  = nn.Parameter(torch.tensor(self._inv_softplus(init_brand_bonus)))
        self.style_bonus_raw  = nn.Parameter(torch.tensor(self._inv_softplus(init_style_bonus)))
        self.liters_bonus_raw = nn.Parameter(torch.tensor(self._inv_softplus(init_liters_bonus)))

    def _inv_softplus(self, x: float) -> float:
        """Raw value such that softplus(raw) ≈ x."""
        return torch.log(torch.expm1(torch.tensor(float(x)))).item()

    def _meta_bonus(
        self,
        category: torch.Tensor,  # (n,) int
        brand: torch.Tensor,     # (n,) int
        style: torch.Tensor,     # (n,) int
        liters: torch.Tensor,    # (n,) float
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            not_self:   (n, n) bool  — True where i ≠ j
            same_cat:   (n, n) bool  — True where category_i == category_j
            meta_bonus: (n, n) float — additive score bonus from metadata
        """
        category = category.to(device)
        brand    = brand.to(device)
        style    = style.to(device)
        liters   = liters.to(device).float()

        n        = category.shape[0]
        not_self = ~torch.eye(n, dtype=torch.bool, device=device)
        same_cat   = category.unsqueeze(0) == category.unsqueeze(1)  # (n, n)
        same_brand = brand.unsqueeze(0)    == brand.unsqueeze(1)     # (n, n)
        same_style = style.unsqueeze(0)    == style.unsqueeze(1)     # (n, n)

        # Liters similarity: sim(i,j) = exp(-gamma * |log(l_i) - log(l_j)|)
        # Closer in log-space (same format/size) → higher similarity → (0, 1]
        eps      = 1e-6
        log_dist = (torch.log(liters.unsqueeze(0) + eps) - torch.log(liters.unsqueeze(1) + eps)).abs()
        sim_liters = torch.exp(-self.gamma * log_dist)  # (n, n)

        bonus = (
            F.softplus(self.brand_bonus_raw)  * same_brand.float()
            + F.softplus(self.style_bonus_raw)  * same_style.float()
            + F.softplus(self.liters_bonus_raw) * sim_liters
        ).masked_fill(~not_self, 0.0)  # no self-edge bonus

        return not_self, same_cat, bonus

    def _build_pairs(
        self,
        mean_scores: torch.Tensor,  # (n, n)
        not_self: torch.Tensor,     # (n, n) bool
        same_cat: torch.Tensor,     # (n, n) bool
    ) -> torch.Tensor:
        """
        Selects top-k directed edges per focal product i.
        Priority: same-category candidates first (strict), then any j neq i (relax).

        Returns:
            pairs: (2, E) with E = n * k_eff
        """
        n     = mean_scores.shape[0]
        k_eff = min(self.k, n - 1)
        dev   = mean_scores.device

        if k_eff == 0:
            return torch.empty(2, 0, dtype=torch.long, device=dev)

        # ── Priority bias trick ────────────────────────────────────────────────
        # To enforce strict: we add a large additive constant (1e6) 
        # to the scores of same-category candidates.
        # This guarantees that any strict candidate always outranks any fallback
        # candidate in the topk, regardless of their actual attention scores.
        # If a row has fewer strict candidates than k_eff, the remaining slots are
        # filled automatically by the best fallback candidates — no conditional
        # logic needed. The bias only affects ranking, not the attention weights
        # computed in Stage 2 (which uses the original unbiased scores).
        if self.use_same_category_strict:
            # (n, n) float: 1e6 where same-category and not self-edge, else 0.
            # 1e6 safely dominates real attention scores (typically in [-10, +10]).
            priority_bias = (same_cat & not_self).float() * 1e6
        else:
            # No priority: all non-self candidates are treated equally.
            priority_bias = torch.zeros(n, n, device=dev)

        # Apply bias and mask self-edges with -inf so they are never selected.
        biased = mean_scores + priority_bias                          # (n, n)
        biased = biased.masked_fill(~not_self, float("-inf"))        # (n, n)

        # ── Single topk on GPU ────────────────────────────────────────────────
        # torch.topk selects the k_eff best columns per row entirely on the GPU.
        # Each row independently enforces strict → fallback via the bias:
        #   - rows with ≥ k_eff strict candidates: top-k are all strict.
        #   - rows with < k_eff strict candidates: strict slots filled first,
        #     remaining slots filled by best fallback candidates.
        _, top_j = torch.topk(biased, k_eff, dim=1)                  # (n, k_eff)

        # ── Build (2, E) edge index tensor entirely on GPU ────────────────────
        # i_idx: each focal product i repeated k_eff times → [0,0,..,1,1,..,n-1,..]
        # j_idx: the k_eff selected neighbors for each i, flattened.
        i_idx = torch.arange(n, device=dev).unsqueeze(1).expand(n, k_eff).reshape(-1)  # (E,)
        j_idx = top_j.reshape(-1)                                                        # (E,)

        return torch.stack([i_idx, j_idx], dim=0)  # (2, E), E = n * k_eff

    def run(
        self,
        h: torch.Tensor,         # (B, n, d_hidden)
        category: torch.Tensor,  # (n,) int
        brand: torch.Tensor,     # (n,) int
        style: torch.Tensor,     # (n,) int
        liters: torch.Tensor,    # (n,) float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n, _ = h.shape
        device  = h.device

        if n <= 1:
            return (torch.empty(2, 0, dtype=torch.long, device=device),
                    torch.empty(B, 0, dtype=h.dtype,    device=device))

        Q = self.q_proj(h)  # (B, n, d_attn)
        K = self.k_proj(h)  # (B, n, d_attn)

        not_self, same_cat, meta_bonus = self._meta_bonus(category, brand, style, liters, device)

        # scores[b, i, j] = q_i · k_j / sqrt(d_attn) + metadata_bonus(i,j)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, n, n)
        scores = scores + meta_bonus.unsqueeze(0)
        scores = scores.masked_fill(~not_self.unsqueeze(0), float("-inf"))

        # ── Stage 1: structural graph (batch-shared) ──────────────────────────
        pairs = self._build_pairs(scores.mean(dim=0), not_self, same_cat)  # (2, E)

        if pairs.numel() == 0:
            return pairs, torch.empty(B, 0, dtype=h.dtype, device=device)

        i_idx, j_idx = pairs[0], pairs[1]
        E = i_idx.numel()

        if E % n != 0:
            raise RuntimeError(f"SparseNeighborSelector: E={E} not divisible by n={n}")

        # ── Stage 2: per-sample soft weights ──────────────────────────────────
        k_eff = E // n
        # Reshape to (B, n, k_eff) to apply softmax per focal product i
        edge_logits  = scores[:, i_idx, j_idx].view(B, n, k_eff)  # (B, n, k_eff)
        edge_weights = F.softmax(edge_logits, dim=-1).reshape(B, E)  # (B, E)

        return pairs, edge_weights  # (2, E), (B, E)

    def forward(self, *args, **kwargs):
        return self.run(*args, **kwargs)