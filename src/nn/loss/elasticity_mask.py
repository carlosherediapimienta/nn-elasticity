import torch

def elasticity_entry_mask(
    obs_mask: torch.Tensor,
    pairs: torch.Tensor | None = None,
    availability: torch.Tensor | None = None,
    price_observed: torch.Tensor | None = None,
    include_diag: bool = True,
) -> torch.Tensor:
    """M[b,i,j] = 1 iff E_{ij} is a valid elasticity for penalty / score / export.

    Target i: demand observed and own price observed (and spline B_i(u_i)).
    Source j: price observed and product available. Demand of j is not required.
    """
    obs = obs_mask.bool()
    pobs = price_observed.bool() if price_observed is not None else torch.ones_like(obs)
    avail = availability.bool() if availability is not None else torch.ones_like(obs)

    valid_target = obs & pobs
    valid_source = pobs & avail

    n = obs.shape[1]
    active = torch.zeros(n, n, dtype=torch.bool, device=obs.device)
    if include_diag:
        d = torch.arange(n, device=obs.device)
        active[d, d] = True
    if pairs is not None and pairs.numel() > 0:
        active[pairs[0], pairs[1]] = True

    return (
        valid_target.unsqueeze(2)
        & valid_source.unsqueeze(1)
        & active.unsqueeze(0)
    )