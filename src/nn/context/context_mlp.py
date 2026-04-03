import torch
import torch.nn as nn

class ContextMLP(nn.Module):
    """
    MLP that encodes the raw context vector into a richer, more expressive representation.

    The concatenated context (store embedding + time + promo + per-product features) contains
    no interactions between its components. This MLP learns non-linear combinations across all
    features, allowing the model to capture patterns such as "high lag_1 + on_promo in store X".

    Smooth activations (tanh, softplus, gelu) are used instead of ReLU to ensure the output
    is differentiable everywhere — a requirement for integration over price.

    Public API: forward() (following nn.Module convention).
    """
    
    def __init__(self, d_in: int, hidden=(256, 128, 64), act="tanh", dropout=0.0):
        """
        Args:
            d_in: dimension of input
            hidden: tuple with dimensions of hidden layers
            act: activation function ('tanh', 'softplus', 'gelu')
            dropout: dropout rate between layers (0.0 = no dropout)
        """
        super().__init__()
        
        # Dictionary of supported activation functions.
        acts = {
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "gelu": nn.GELU(),
        }
        
        # We check that the activation function is supported.
        if act not in acts:
            raise ValueError(f"Activation '{act}' not supported. Use: {list(acts.keys())}")
        
        # We get the activation function.
        a = acts[act]

        # We build the MLP. Architecture: [Linear, Activation, Dropout] * num_layers.
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(a)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        # We set the output dimension of the MLP.
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Process context through the MLP.
        
        Args:
            c: (B, d_in) context tensor
        
        Returns:
            (B, out_dim) processed representation. This is called h and
            named latent representation in the article.
        """
        return self.net(c)