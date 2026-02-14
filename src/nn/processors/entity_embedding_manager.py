import torch
import torch.nn as nn


class EntityEmbeddingManager(nn.Module):
    """
    Manage embeddings of entities: store, UPC, week.
    API pública: run().
    """
    
    def __init__(
        self,
        n_stores: int,
        n_upcs: int,
        n_weeks: int,
        d_store: int = 24,
        d_upc: int = 48,
        d_week: int = 12
    ):
        """
        Args:
            n_stores: number of unique stores
            n_upcs: number of unique UPCs
            n_weeks: number of unique weeks
            d_store: dimension of store embedding
            d_upc: dimension of UPC embedding
            d_week: dimension of week embedding
        """
        super().__init__()
        self.emb_store = nn.Embedding(n_stores, d_store)
        self.emb_upc = nn.Embedding(n_upcs, d_upc)
        self.emb_week = nn.Embedding(n_weeks, d_week)
        
        self.d_store = d_store
        self.d_upc = d_upc
        self.d_week = d_week
    
    @property
    def out_dim(self) -> int:
        """Total dimension of concatenated embeddings."""
        return self.d_store + self.d_upc + self.d_week
    
    def run(
        self,
        store_code: torch.Tensor,
        upc_code: torch.Tensor,
        week_id: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Generate embeddings for store, UPC and week.
        
        Args:
            store_code: (B,) tensor with store indices
            upc_code: (B,) tensor with UPC indices
            week_id: (B,) tensor with week indices
        
        Returns:
            dict with:
                'e_store': (B, d_store) store embedding
                'e_upc': (B, d_upc) UPC embedding
                'e_week': (B, d_week) week embedding
                'concat': (B, out_dim) concatenated embeddings
        """
        es = self.emb_store(store_code.long())
        ei = self.emb_upc(upc_code.long())
        et = self.emb_week(week_id.long())
        
        return {
            'e_store': es,
            'e_upc': ei,
            'e_week': et,
            'concat': torch.cat([es, ei, et], dim=1)
        }
    
    def forward(self, *args, **kwargs):
        """Alias for compatibility with nn.Module."""
        return self.run(*args, **kwargs)