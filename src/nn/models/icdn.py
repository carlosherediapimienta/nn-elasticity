import torch
import torch.nn as nn
from ..spline import CubicSplineBasis
from .integrable_demand_head import IntegrableDemandHead


class ICDN(nn.Module):
    """
    Integrable Context-Dependent Demand Network 1D.
    
    Complete model that integrates:
      1. context_builder: generate context vector c
      2. price_spline: calculate spline bases and derivatives for price
      3. head: predict demand and elasticity
    
    Pipeline:
      c = context_builder(store_code, upc_code, week_id, on_promo, promo_B, promo_C, promo_S, liters_per_upc)
      (Bx, dBx, ddBx) = price_spline(log_price)
      y_hat, eps_hat = head(c, log_price, Bx, dBx)    
    """
    
    def __init__(self, context_builder, price_splines: nn.ModuleList, head, n: int):
        """
        Args:
            context_builder: module that generates context vector
                            (e.g.: DemandContextEmbeddings)
            price_spline: generator of spline bases for price
            head: head that predicts demand and elasticity
        """
        super().__init__()
        self.context_builder = context_builder
        self.price_splines = price_splines
        self.head = head
        self.n = n


    def run(self, batch, return_parts=False):
        c = self.context_builder(batch)
        x = torch.stack([batch[f"log_price_{i}"] for i in range(self.n)], dim=1)

        Bx_list, dBx_list, ddBx_list, IBx_list = [], [], [], []
        for i, spline in enumerate(self.price_splines):
            Bx_i, dBx_i, ddBx_i, IBx_i = spline(x[:, i])   # ← ahora 4 salidas
            Bx_list.append(Bx_i)
            dBx_list.append(dBx_i)
            ddBx_list.append(ddBx_i)
            IBx_list.append(IBx_i)

        Bx   = torch.stack(Bx_list,   dim=1)
        dBx  = torch.stack(dBx_list,  dim=1)
        ddBx = torch.stack(ddBx_list, dim=1)
        IBx  = torch.stack(IBx_list,  dim=1)   # ← nuevo (B, n, K)

        y_hat, eps_hat, aux = self.head.run(c, x, Bx, dBx, ddBx, IBx)

        if return_parts:
            aux.update({"c": c, "Bx": Bx, "dBx": dBx, "ddBx": ddBx, "IBx": IBx})
            return y_hat, eps_hat, aux

        return y_hat, eps_hat
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module's forward method."""
        return self.run(*args, **kwargs)