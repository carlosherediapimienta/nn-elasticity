import torch
import torch.nn as nn
from .elasticity_first_predictor import ElasticityFirstPredictor

class BaselineModel(nn.Module):
    """
    Full elasticity-first model. Mirror of ICDN_1D for the baseline.
    Unpacks the batch, builds the context c and calls the predictor.
    Public API: forward()
    """

    def __init__(self, context_builder: nn.Module, predictor: ElasticityFirstPredictor):
        super().__init__()
        self.context_builder = context_builder
        self.predictor = predictor

        self.lag_keys = [
            "lag_y_1", "lag_y_2", "lag_y_4",
            "rolling_mean_y_4", "rolling_mean_y_13",
            "lag_y_1_missing", "lag_y_2_missing", "lag_y_4_missing",
            "rolling_mean_y_4_missing", "rolling_mean_y_13_missing",
            "lag_x_1", "delta_x_1", "rolling_mean_x_4",
            "lag_onpromo_1", "lag_onpromo_2", "weeks_since_promo", "week_gap_1",
        ]

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: dict with keys of DemandDataset

        Returns:
            y_hat: (B,)
            x_nd:  (B, 1)  log-price as 2D tensor for the integrator
            c:     (B, d)  vector of context
        """
        lag_features = torch.stack([batch[k] for k in self.lag_keys], dim=1)

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
            return_parts=False,
        )

        x_nd = batch["log_price_per_liter"].unsqueeze(-1)  # (B,) → (B, 1)
        y_hat = self.predictor(x_nd, c)                    # (B, 1)
        return y_hat.squeeze(-1), x_nd, c                  # (B,), (B,1), (B,d)