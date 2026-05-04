from torch.utils.data import DataLoader, Dataset
import math


class DataLoaderFactory:
    """
    Factory to create configured DataLoaders.
    Public API: create_train_loader(), create_eval_loader().
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int | None = 2,
    ):
        """
        Args:
            batch_size: default batch size
            num_workers: number of workers for data loading
            pin_memory: if True, uses pinned memory (faster on GPU)
        """
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        # Why this configuration?
        # 1. Persistent workers: If True, the workers will be persistent and the data will be loaded in a streaming fashion.
        # 2. Prefetch factor: The number of batches to prefetch from the dataset.
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor 
    
    def create_train_loader(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader for training.
        
        Args:
            dataset: Dataset to load
            batch_size: batch size
            shuffle: if True, shuffles the data
            drop_last: if True, drops the last incomplete batch
        
        Returns:
            Configured DataLoader for training
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )
    
    def create_eval_loader(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create DataLoader for evaluation.
        
        Args:
            dataset: Dataset to load
            batch_size: batch size
            shuffle: if True, shuffles the data (normally False for eval)
        
        Returns:
            Configured DataLoader for evaluation
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False, # Don't drop the last batch
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )