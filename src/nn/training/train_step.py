import torch
import torch.nn as nn
from .batch_processor import BatchProcessor
from .gradient_handler import GradientHandler


class TrainStepExecutor:
    """
    Execute a complete training step.
    Public API: run().
    """
    
    def __init__(
        self,
        device: str = "cuda",
        max_grad_norm: float | None = None,
        batch_processor: BatchProcessor | None = None,
        gradient_handler: GradientHandler | None = None
    ):
        """
        Args:
            device: computation device
            max_grad_norm: maximum gradient norm for clipping
            batch_processor: custom batch processor (optional)
            gradient_handler: custom gradient handler (optional)
        """
        self.device = device
        self.batch_processor = batch_processor or BatchProcessor(device=device)
        self.gradient_handler = gradient_handler or GradientHandler(max_grad_norm=max_grad_norm)
    
    def run(
        self,
        model: nn.Module,
        batch: dict,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Execute training step.
        
        Args:
            model: model to train
            batch: dict with data
            loss_fn: loss function
            optimizer: optimizer
            scaler: GradScaler for AMP (optional)
        
        Returns:
            logs: dict with metrics of the step
        """
        model.train()
        
        # 1. Prepare batch
        batch = self.batch_processor.prepare_batch(batch)
        y_true = self.batch_processor.extract_target(batch)
        
        # 2. Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # 3. Forward pass (with AMP if applicable)
        use_amp = scaler is not None
        with torch.amp.autocast(device_type=self.device, enabled=use_amp):
            y_hat, eps_hat, aux = model(batch, return_parts=True)
            y_hat = y_hat.float().view(-1)
            eps_hat = eps_hat.float().view(-1)
            
            # Calculate loss
            loss, logs = loss_fn(y_hat, y_true, eps_hat, aux["w"], aux["ddBx"])
        
        # 4. Backward pass
        self.gradient_handler.backward(loss, scaler)
        
        # 5. Gradient clipping
        self.gradient_handler.clip_gradients(model, optimizer, scaler)
        
        # 6. Optimizer step
        self.gradient_handler.step(optimizer, scaler)
        
        return logs