import torch
import torch.nn as nn

class ICDN(nn.Module):
    """
    Integrable Context-Dependent Demand Network (multiproduct).
    Combines three submodules into a single forward pass:
      1. context_builder  (MultiProductContextEmbeddings): encodes store, UPC,
         week and promotional features into a context vector c.
      2. price_splines    (MultiCubicSplineBasis): evaluates cubic spline bases
         and their derivatives / antiderivatives for all n log-prices at once.
      3. head             (IntegrableDemandHead): maps (c, x, splines) to
         predicted log-demand y_hat and own-price elasticity eps_hat.
    Pipeline:
      c                            = context_builder(batch)          # (B, context_dim)
      x                            = stack(log_price_0..n-1)         # (B, n)
      Bx, dBx                      = price_splines(x)               # (B, n, K) each
      y_hat, eps_hat               = head(c, x, Bx, dBx, alpha, u, pairs) # (B, n) each
    Args:
        context_builder (nn.Module): produces the context vector c.
        price_splines   (MultiCubicSplineBasis): vectorized spline evaluator.
        head            (IntegrableDemandHead): demand and elasticity predictor.
        n               (int): number of products.
    """
    
    def __init__(self, 
        context_builder,
        price_splines,
        head,
        n,
    ):
        """
        Args:
            context_builder: MultiProductContextEmbeddings
                            (e.g.: DemandContextEmbeddings)
            price_spline: MultiCubicSplineBasis
            head: IntegrableDemandHead
        """
        super().__init__()
        self.context_builder = context_builder # MultiProductContextEmbeddings
        self.price_splines = price_splines # MultiCubicSplineBasis
        self.head = head # IntegrableDemandHead
        self.n = n # Number of products

    def run(self, batch, return_parts: bool = False, compute_E: bool = False, neighbor_meta: dict[str, torch.Tensor] | None = None):
        # (B, n, d_token) - Context vector.
        tokens = self.context_builder(batch)
        # (B, n) - Price vector.
        # batch["prices"] is already (B, n) — pre-stacked in MultiProductDataset.__init__.
        x = batch["prices"]
        # Compute the spline bases, derivatives, and antiderivatives.
        Bx, dBx, ddBx = self.price_splines(x)
        # Compute the predicted demand and elasticity.
        availability = batch["availability"]
        y_hat, eps_hat, aux = self.head.run(
            tokens=tokens,
            x=x,
            Bx=Bx,
            dBx=dBx,
            return_E=compute_E,
            neighbor_meta=neighbor_meta,
            availability=availability,
        )

        if return_parts:
            aux.update({"tokens": tokens, "Bx": Bx, "dBx": dBx, "ddBx": ddBx})
            return y_hat, eps_hat, aux

        return y_hat, eps_hat
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module's forward method."""
        return self.run(*args, **kwargs)