import torch


class PromoFeatureProcessor:
    """
    Process features of promotion in vectorial format.
    API pública: run().
    """
    
    def run(
        self,
        on_promo: torch.Tensor,
        promo_B: torch.Tensor,
        promo_C: torch.Tensor,
        promo_S: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert promotion flags into a vectorial tensor.
        
        Args:
            on_promo: (B,) tensor bool/int/float indicating if there is promotion
            promo_B: (B,) tensor for promotion type B
            promo_C: (B,) tensor for promotion type C
            promo_S: (B,) tensor for promotion type S
        
        Returns:
            (B, 4) tensor with [on_promo, promo_B, promo_C, promo_S]
        """
        return torch.stack([on_promo, promo_B, promo_C, promo_S], dim=1).float()