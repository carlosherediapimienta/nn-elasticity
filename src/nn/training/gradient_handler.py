import torch
import torch.nn as nn


class GradientHandler:
    """
    Manages gradient clipping and AMP scaling.
    Public API: backward(), step().
    """
    
    def __init__(self, max_grad_norm: float | None = None):
        """
        Args:
            max_grad_norm: maximum gradient norm for clipping (None = no clipping)
        """
        self.max_grad_norm = max_grad_norm
    
    def backward(
        self,
        loss: torch.Tensor,
        scaler: torch.cuda.amp.GradScaler | None = None
    ) -> None:
        """
        Execute backward pass.
        
        Args:
            loss: loss tensor
            scaler: GradScaler for AMP (optional)
        """
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None = None
    ) -> None:
        """
        Apply gradient clipping if configured.
        
        Args:
            model: model with parameters
            optimizer: optimizer
            scaler: GradScaler for AMP (optional)
        """
        if self.max_grad_norm is not None:
            if scaler is not None:
                scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
    
    def step(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None = None
    ) -> None:
        """
        Execute optimization step.
        
        Args:
            optimizer: optimizer
            scaler: GradScaler for AMP (optional)
        """
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()