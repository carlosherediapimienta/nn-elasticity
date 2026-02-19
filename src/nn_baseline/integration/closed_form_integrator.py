import torch


class ClosedFormIntegrator:
    """
    Integración closed-form para ConstantMatrixElasticity.
    y = y0 + W(x - x0)
    Public API: run()
    """

    def run(
        self,
        W: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor,
        y0: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            W:  (n, n) matriz de elasticidad constante
            x:  (B, n) log-precios destino
            x0: (B, n) log-precios origen
            y0: (B, n) demanda en el punto origen

        Returns:
            y: (B, n) demanda predicha
        """
        dx = x - x0
        return y0 + torch.einsum("ij,bj->bi", W, dx)