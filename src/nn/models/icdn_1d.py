import torch
import torch.nn as nn
from ..spline import CubicSplineBasis
from .integrable_demand_head import IntegrableDemandHead1D


class ICDN_1D(nn.Module):
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
    
    def __init__(
        self,
        context_builder: nn.Module,
        price_spline: CubicSplineBasis,
        head: IntegrableDemandHead1D
    ):
        """
        Args:
            context_builder: module that generates context vector
                            (e.g.: DemandContextEmbeddings)
            price_spline: generator of spline bases for price
            head: head that predicts demand and elasticity
        """
        super().__init__()
        self.context_builder = context_builder
        self.price_spline = price_spline
        self.head = head
        self.lag_keys = [
            "lag_y_1", "lag_y_2", "lag_y_4",
            "rolling_mean_y_4", "rolling_mean_y_13",
            "lag_y_1_missing", "lag_y_2_missing", "lag_y_4_missing",
            "rolling_mean_y_4_missing", "rolling_mean_y_13_missing",
            "lag_x_1", "delta_x_1", "rolling_mean_x_4",
            "lag_onpromo_1", "lag_onpromo_2", "weeks_since_promo",
            "week_gap_1",
        ]

    def run(self, batch: dict, return_parts: bool = False):
        """
        Public API. Prediction of demand and elasticity.
        
        Args:
            batch: dict with keys:
                - store_code: (B,)
                - upc_code: (B,)
                - week_id: (B,)
                - on_promo: (B,)
                - promo_B/C/S: (B,)
                - liters_per_upc: (B,)
                - log_price_per_liter: (B,)
            return_parts: if True, return intermediate components
        
        Returns:
            If return_parts=False:
                y_hat: (B,) predicted demand
                eps_hat: (B,) predicted price elasticity
            
            If return_parts=True:
                y_hat: (B,)
                eps_hat: (B,)
                aux: dict with keys {'b', 'beta', 'w', 'c', 'Bx', 'dBx', 'ddBx'}
        """

        lag_features = torch.stack([batch[k] for k in self.lag_keys], dim=1)

        # 1) Build context vector c
        c = self.context_builder(
            batch["store_code"],
            batch["upc_code"],
            batch["week_id"],
            batch["on_promo"],
            batch["promo_B"],
            batch["promo_C"],
            batch["promo_S"],
            batch["liters_per_upc"],
            lag_features,
            return_parts=False
        )

        # 2) Calculate spline bases for price x
        x = batch["log_price_per_liter"]
        Bx, dBx, ddBx = self.price_spline(x)

        # 3) Calculate demand and elasticity
        y_hat, eps_hat, aux = self.head.run(c, x, Bx, dBx)

        if return_parts:
            aux.update({
                "c": c,
                "Bx": Bx,
                "dBx": dBx,
                "ddBx": ddBx
            })
            return y_hat, eps_hat, aux
        
        return y_hat, eps_hat
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module's forward method."""
        return self.run(*args, **kwargs)