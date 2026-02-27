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
        pin_memory: bool = True
    ):
        """
        Args:
            batch_size: default batch size
            num_workers: number of workers for data loading
            pin_memory: if True, uses pinned memory (faster on GPU)
        """
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _calculate_batch_size(self, dataset: Dataset, target_batches: int = 10) -> int:
        raw = max(1, len(dataset) // target_batches)
        return max(1, 2 ** round(math.log2(raw)))
    
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
            batch_size: batch size (uses default if None)
            shuffle: if True, shuffles the data
            drop_last: if True, drops the last incomplete batch
        
        Returns:
            Configured DataLoader for training
        """

        if batch_size is None:
            batch_size = self._calculate_batch_size(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last
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
            batch_size: batch size (uses default if None)
            shuffle: if True, shuffles the data (normally False for eval)
        
        Returns:
            Configured DataLoader for evaluation
        """
        if batch_size is None:
            batch_size = self._calculate_batch_size(dataset)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False  # Don't drop in eval
        )