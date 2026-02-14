import torch

class BatchProcessor:
    """
    Process batches: device transfer, extraction of targets.
    Public API: prepare_batch(), extract_target().
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: destination device ('cuda', 'cpu', etc.)
        """
        self.device = device
    
    def prepare_batch(self, batch: dict) -> dict:
        """
        Move batch to device.
        
        Args:
            batch: dict with tensors
        
        Returns:
            batch on device
        """
        return {k: v.to(self.device) for k, v in batch.items()}
    
    def extract_target(self, batch: dict, target_key: str = "log_liters_sold") -> torch.Tensor:
        """
        Extract target tensor and format it.
        
        Args:
            batch: dict with data
            target_key: key of the target in batch
        
        Returns:
            y_true: (B,) target tensor
        """
        return batch[target_key].float().view(-1)