import torch
import torch.nn as nn
from .processors import (
    EntityEmbeddingManager,
    PromoFeatureProcessor,
    LiterFeatureProcessor
)
from .time_features import FourierTimeFeatures


class DemandContextEmbeddings(nn.Module):
    """
    Orchestrate the construction of c_{s,i,t} = concat(e_s, e_i, e_t, Fourier(t), p, z).
    Public API: run().
    
    Delegation:
    - EntityEmbeddingManager: embeddings of entities (store, UPC, week)
    - FourierTimeFeatures: temporal features (Fourier + trend)
    - PromoFeatureProcessor: promotion features (4 features)
    - LiterFeatureProcessor: log transform of liters (1 feature)
    
    Assumption: store_code/upc_code/week_id have been reindexed to [0..N-1]
    to be valid indices for nn.Embedding.
    """
    
    def __init__(
        self,
        n_stores: int,
        n_upcs: int,
        n_weeks: int,
        n_lags: int = 17,
        d_store: int = 24,
        d_upc: int = 48,
        d_week: int = 12,
        fourier_period: float = 52.0,
        fourier_harmonics: int = 4,
        include_trend: bool = True,
        week_min: float | None = None,
        week_max: float | None = None,
        log_liters_eps: float = 1e-6,
        # Opcional: inyección de dependencias para testing
        entity_embedder: EntityEmbeddingManager | None = None,
        time_features: FourierTimeFeatures | None = None,
        promo_processor: PromoFeatureProcessor | None = None,
        liter_processor: LiterFeatureProcessor | None = None
    ):
        """
        Args:
            n_stores: number of unique stores
            n_upcs: number of unique UPCs
            n_weeks: number of unique weeks
            n_lags: number of lag features
            d_store: dimension of store embedding
            d_upc: dimension of UPC embedding
            d_week: dimension of week embedding
            fourier_period: period for Fourier features (default: 52 weeks)
            fourier_harmonics: number of harmonics
            include_trend: include normalized trend
            week_min/week_max: range for normalizing trend
            log_liters_eps: epsilon to avoid log(0)
            entity_embedder: custom EntityEmbeddingManager (optional)
            time_features: custom FourierTimeFeatures (optional)
            promo_processor: custom PromoFeatureProcessor (optional)
            liter_processor: custom LiterFeatureProcessor (optional)
        """
        super().__init__()
        
        # Dependency injection or default creation
        self.entity_embedder = entity_embedder or EntityEmbeddingManager(
            n_stores=n_stores,
            n_upcs=n_upcs,
            n_weeks=n_weeks,
            d_store=d_store,
            d_upc=d_upc,
            d_week=d_week
        )
        
        self.time_features = time_features or FourierTimeFeatures(
            period=fourier_period,
            harmonics=fourier_harmonics,
            include_trend=include_trend
        )
        
        self.promo_processor = promo_processor or PromoFeatureProcessor()
        self.liter_processor = liter_processor or LiterFeatureProcessor(eps=log_liters_eps)
        
        self.n_lags = n_lags
        self.week_min = week_min
        self.week_max = week_max
    
    @property
    def out_dim(self) -> int:
        """Total dimension of the context vector."""
        return (
            self.entity_embedder.out_dim +  # embeddings (store + UPC + week)
            self.time_features.out_dim +    # Fourier + trend
            4 +                              # promo features (on_promo, B, C, S)
            1 +                              # log_liters
            self.n_lags                   # lag features
        )
    
    def run(
        self,
        store_code: torch.Tensor,
        upc_code: torch.Tensor,
        week_id: torch.Tensor,
        on_promo: torch.Tensor,
        promo_B: torch.Tensor,
        promo_C: torch.Tensor,
        promo_S: torch.Tensor,
        liters_per_upc: torch.Tensor,
        lag_features: torch.Tensor,
        return_parts: bool = False
    ):
        """
        Public API. Build the complete context vector.
        
        Args:
            store_code: (B,) store indices [0..n_stores-1]
            upc_code: (B,) UPC indices [0..n_upcs-1]
            week_id: (B,) week indices [0..n_weeks-1]
            on_promo: (B,) general promotion flag
            promo_B: (B,) promotion type B flag
            promo_C: (C,) promotion type C flag
            promo_S: (S,) promotion type S flag
            liters_per_upc: (B,) liters per UPC (float)
            lag_features: (B, n_lag_features) lag features
            return_parts: if True, return dict with individual components
        
        Returns:
            If return_parts=False: (B, out_dim) tensor with context vector
            If return_parts=True: dict with:
                'c': (B, out_dim) complete vector
                'e_store': (B, d_store)
                'e_upc': (B, d_upc)
                'e_week': (B, d_week)
                'fourier_t': (B, d_time) time features
                'promo4': (B, 4) promotion features
                'log_liters_per_upc': (B, 1) log liters per UPC
        """
        # 1. Entity embeddings (store, UPC, week)
        emb_result = self.entity_embedder.run(store_code, upc_code, week_id)
        
        # 2. Time features (Fourier + trend)
        ft = self.time_features(
            week_id,
            week_min=self.week_min,
            week_max=self.week_max
        )
        
        # 3. Promo features (4-vector)
        p = self.promo_processor.run(on_promo, promo_B, promo_C, promo_S)
        
        # 4. Liter features (log transform)
        z = self.liter_processor.run(liters_per_upc)
        
        # 5. Concatenación final: [e_s, e_i, e_t, Fourier(t), p, z]
        parts = [emb_result['concat'], ft, p, z, lag_features]
        c = torch.cat(parts, dim=1)
        
        if return_parts:
            return {
                "c": c,
                "e_store": emb_result['e_store'],
                "e_upc": emb_result['e_upc'],
                "e_week": emb_result['e_week'],
                "fourier_t": ft,
                "promo4": p,
                "log_liters_per_upc": z,
                "lag_features": lag_features,
            }
        return c
    
    def forward(self, *args, **kwargs):
        """Alias de run() para compatibilidad con nn.Module."""
        return self.run(*args, **kwargs)