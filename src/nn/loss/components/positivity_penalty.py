import torch
import torch.nn.functional as F

class PositivityPenalty:
    """
    Positivity penalty: mean(ReLU(eps_hat)).
    Promotes negative elasticities (downward sloping demand).
    Public API: run().
    """
    
    def run(self, eps_hat: torch.Tensor) -> torch.Tensor:
        """
        Calculate positivity penalty.
        
        Args:
            eps_hat: (B,) predicted elasticities
        
        Returns:
            penalty: scalar, mean(ReLU(eps_hat))
        """
        return torch.mean(F.relu(eps_hat))