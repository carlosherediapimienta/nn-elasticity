import torch
import torch.nn as nn

class SharedProductEncoder(nn.Module):
    """
    MLP that encodes per-product context tokens into richer representations.
    Input tokens have shape (B, n, d_token). This encoder is applied independently
    to each product token (shared weights across products). It learns non-linear
    feature combinations within each token (e.g., "high lag_1 + on_promo in store X"),
    but does not model product-to-product interactions by itself.
    Smooth activations (tanh, softplus, gelu) are used instead of ReLU to keep
    the mapping differentiable everywhere, which is useful for price-integration
    constraints in the demand model.
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process context through the MLP.
        
        Args:
            tokens: (B, n, d_token) tokens tensor
        
        Returns:
            (B, n, out_dim) latent representation. This is called h and
            named latent representation in the article.
        """
        B, n, d_token = tokens.shape
        flat = tokens.view(B*n, d_token) # (B*n, d_token)
        h = self.net(flat) # (B*n, out_dim)
        # This procedure is equivalent to:
        # for i in products: h_i = MLP(token_i) but vectorized.
        # Therefore, for each n, we get h_n but in representation (B, n, out_dim),
        # where out_dim is the dimension of the latent representation.
        return h.view(B, n, self.out_dim) # (B, n, out_dim)