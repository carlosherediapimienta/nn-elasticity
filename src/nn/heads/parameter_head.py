import torch
import torch.nn as nn
import torch.nn.functional as F


class DemandParameterHead(nn.Module):
    """
    Generate parameters b(c), beta(c), w(c) from context representation.
    
    Parameters:
    - b: intercept term
    - beta: linear price coefficient
    - w: spline weights
    
    API pública: run().
    """
    
    def __init__(
        self,
        hidden_dim: int,
        K_splines: int,
        enforce_negative_beta: bool = False
    ):
        """
        Args:
            hidden_dim: dimension of input (output of context encoder)
            K_splines: number of spline bases (dimension of w)
            enforce_negative_beta: if True, beta will always be <= 0 (downward sloping)
        """
        super().__init__()
        self.head_b = nn.Linear(hidden_dim, 1)
        self.head_beta = nn.Linear(hidden_dim, 1)
        self.head_w = nn.Linear(hidden_dim, K_splines)
        
        self.enforce_negative_beta = enforce_negative_beta
    
    def run(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generate parameters from context representation.
        
        Args:
            h: (B, hidden_dim) processed context representation
        
        Returns:
            dict with:
                'b': (B,) intercept term
                'beta': (B,) linear coefficient
                'w': (B, K_splines) spline weights
        """
        b = self.head_b(h).squeeze(-1)  # (B,)
        beta_raw = self.head_beta(h).squeeze(-1)  # (B,)
        
        if self.enforce_negative_beta:
            beta = -F.softplus(beta_raw)  # always <= 0
        else:
            beta = beta_raw
        
        w = self.head_w(h)  # (B, K_splines)
        
        return {
            'b': b,
            'beta': beta,
            'w': w
        }
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)