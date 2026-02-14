import torch
import torch.nn as nn
from .batch_processor import BatchProcessor
from ..loss.metrics import RegressionMetrics


class EvalStepExecutor:
    """
    Execute a complete evaluation step.
    Public API: run().
    """
    
    def __init__(
        self,
        device: str = "cuda",
        batch_processor: BatchProcessor | None = None,
        metrics_calc: RegressionMetrics | None = None
    ):
        """
        Args:
            device: computation device
            batch_processor: custom batch processor (optional)
            metrics_calc: custom metrics calculator (optional)
        """
        self.batch_processor = batch_processor or BatchProcessor(device=device)
        self.metrics_calc = metrics_calc or RegressionMetrics()
    
    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: nn.Module
    ) -> dict[str, torch.Tensor]:
        """
        Execute evaluation step.
        
        Args:
            model: model to evaluate
            batch: dict with data
            loss_fn: loss function
        
        Returns:
            logs: dict with metrics of the step
        """
        model.eval()
        
        # 1. Prepare batch
        batch = self.batch_processor.prepare_batch(batch)
        y_true = self.batch_processor.extract_target(batch)
        
        # 2. Forward pass
        y_hat, eps_hat, aux = model(batch, return_parts=True)
        y_hat = y_hat.float().view(-1)
        eps_hat = eps_hat.float().view(-1)
        
        # 3. Calculate loss
        loss, logs = loss_fn(y_hat, y_true, eps_hat, aux["w"], aux["ddBx"])
        
        # 4. Additional metrics
        extra_metrics = self.metrics_calc.run(y_hat, y_true)
        logs.update(extra_metrics)
        
        return logs