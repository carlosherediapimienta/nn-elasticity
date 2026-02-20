import torch
import torch.nn as nn
from .elasticity_first_predictor import ElasticityFirstPredictor

class BaselineModel(nn.Module):

    def __init__(self, context_builder: nn.Module, predictor: ElasticityFirstPredictor, n: int):
        super().__init__()
        self.context_builder = context_builder
        self.predictor = predictor
        self.n = n

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.context_builder(batch)                              # (B, d_context)

        x = torch.stack(
            [batch[f"log_price_{i}"] for i in range(self.n)], dim=1
        )                                                            # (B, n)

        y_hat = self.predictor(x, c)                                 # (B, n)
        return y_hat, x, c