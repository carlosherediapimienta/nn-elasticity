import torch


class LiterFeatureProcessor:
    """
    Transform liters_per_upc to logarithmic scale.
    API pública: run().
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: minimum value to avoid log(0)
        """
        self.eps = float(eps)
    
    def run(self, liters_per_upc: torch.Tensor) -> torch.Tensor:
        """
        Apply logarithmic transformation to liters per UPC.
        
        Args:
            liters_per_upc: (B,) tensor with liters per UPC
        
        Returns:
            (B, 1) tensor with log(liters_per_upc), clamped to avoid log(0)
        """
        return torch.log(liters_per_upc.float().clamp_min(self.eps)).view(-1, 1)