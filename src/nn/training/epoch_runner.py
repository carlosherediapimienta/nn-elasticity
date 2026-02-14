import torch
import torch.nn as nn
from .train_step import TrainStepExecutor
from .eval_step import EvalStepExecutor
from .metric_aggregator import MetricAggregator


class EpochRunner:
    """
    Execute complete training/evaluation epochs.
    Public API: run_train_epoch(), run_eval_epoch().
    """
    
    def __init__(
        self,
        device: str = "cuda",
        max_grad_norm: float | None = 1.0,
        train_executor: TrainStepExecutor | None = None,
        eval_executor: EvalStepExecutor | None = None
    ):
        """
        Args:
            device: computation device
            max_grad_norm: maximum gradient norm for clipping
            train executor: custom train step executor (optional)
            eval executor: custom evaluation step executor (optional)
        """
        self.train_executor = train_executor or TrainStepExecutor(
            device=device,
            max_grad_norm=max_grad_norm
        )
        self.eval_executor = eval_executor or EvalStepExecutor(device=device)
    
    def run_train_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None = None
    ) -> dict[str, float]:
        """
        Execute training epoch.
        
        Args:
            model: model to train
            loader: training DataLoader
            loss_fn: loss function
            optimizer: optimizer
            scaler: GradScaler for AMP (optional)
        
        Returns:
            aggregated metrics of the epoch
        """
        aggregator = MetricAggregator()
        
        for batch in loader:
            logs = self.train_executor.run(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler
            )
            batch_size = batch["log_liters_sold"].shape[0]
            aggregator.add(logs, batch_size)
        
        return aggregator.compute()
    
    @torch.no_grad()
    def run_eval_epoch(
        self,
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module
    ) -> dict[str, float]:
        """
        Execute evaluation epoch.
        
        Args:
            model: model to evaluate
            loader: validation DataLoader
            loss_fn: loss function
        
        Returns:
            aggregated metrics of the epoch
        """
        aggregator = MetricAggregator()
        
        for batch in loader:
            logs = self.eval_executor.run(
                model=model,
                batch=batch,
                loss_fn=loss_fn
            )
            batch_size = batch["log_liters_sold"].shape[0]
            aggregator.add(logs, batch_size)
        
        return aggregator.compute()