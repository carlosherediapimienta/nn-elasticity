import numpy as np
import pandas as pd
from typing import List, Tuple


class TemporalSplitter:
    """Chronological train/validation splits for time-ordered data (no random shuffling)."""

    def __init__(self, week_col: str = "week_id") -> None:
        self.week_col = week_col

    def single_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Single chronological split: first train_frac of weeks = train, rest = val.
        """
        sorted_weeks = sorted(df[self.week_col].unique()) # Sort weeks in ascending order
        threshold_idx = int(len(sorted_weeks) * train_frac)
        week_threshold = sorted_weeks[threshold_idx] # Week threshold for the split

        train_df = df[df[self.week_col] < week_threshold].copy()
        val_df = df[df[self.week_col] >= week_threshold].copy() 

        return train_df, val_df

    def expanding_splits(
        self,
        df: pd.DataFrame,
        n_folds: int,
        min_train_frac: float = 0.5,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Expanding window: each fold trains only with past and validates in future.
        Measures real temporal generalization (no leakage).

        Args:
            df: DataFrame with week_col column.
            n_folds: Number of folds.
            min_train_frac: Minimum fraction of weeks for the first train (e.g. 0.5 = 50%).

        Returns:
            List of (train_df, val_df) with length <= n_folds.
        """
        sorted_weeks = sorted(df[self.week_col].unique()) # Sort weeks in ascending order
        n_weeks = len(sorted_weeks)

        min_train_weeks = max(1, int(n_weeks * min_train_frac)) # Minimum number of weeks for the first train
        remaining_weeks = n_weeks - min_train_weeks # Remaining number of weeks
        val_size = max(1, remaining_weeks // n_folds) # Size of the validation set

        # Loop through the folds, we compute the train and validation sets for each fold
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for i in range(n_folds):
            train_end = min_train_weeks + i * val_size
            val_end = min(train_end + val_size, n_weeks)

            if val_end <= train_end:
                break # If the validation end is before the train end, break the loop

            train_weeks = sorted_weeks[:train_end]
            val_weeks = sorted_weeks[train_end:val_end]

            train_df = df[df[self.week_col].isin(train_weeks)].copy()
            val_df = df[df[self.week_col].isin(val_weeks)].copy()

            if len(val_df) == 0:
                continue

            folds.append((train_df, val_df))

        return folds


class BlockBootstrapSampler:
    """
    Block bootstrap resampler for panel time-series data.
    Instead of sampling individual weeks at random (which would break temporal
    dependencies), this class samples contiguous blocks of `block_size` weeks
    with replacement. This preserves the short-run autocorrelation structure
    within each block while still introducing variability across bootstrap draws.
    Args:
        week_col   (str):                       Name of the week identifier column.
        block_size (int):                       Number of consecutive weeks per block.
        rng        (np.random.Generator|None):  Random number generator for reproducibility.
                                                Defaults to a fresh Generator if not provided.
    Public API:
        sample(df, train_weeks) -> pd.DataFrame
    """

    def __init__(
        self,
        week_col: str = "week_id",
        block_size: int = 4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.week_col = week_col
        self.block_size = block_size
        self.rng = rng or np.random.default_rng()

    def sample(
        self,
        df: pd.DataFrame,
        train_weeks: List | np.ndarray,
    ) -> pd.DataFrame:
        """
        Block bootstrap with replacement: preserves block multiplicity.
        """
        # Sort train_weeks to guarantee block contiguity; convert to list if needed.
        train_weeks = (
            sorted(train_weeks)
            if not isinstance(train_weeks, np.ndarray)
            else np.sort(train_weeks).tolist()
        )
        n_weeks = len(train_weeks) # Number of weeks in the training set.

        if n_weeks < self.block_size: 
            # If the number of weeks is less than the block size, we return the
            # entire dataframe.
            return df[df[self.week_col].isin(train_weeks)].copy()

        # Build non-overlapping block start indices, then sample n_blocks of them
        # with replacement. Each sampled index selects a contiguous window of
        # block_size weeks from train_weeks.
        #
        # Example — 10 weeks, block_size=4:
        #   train_weeks  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #   block_starts = [0, 4, 8]          indices into train_weeks
        #   n_blocks     = 3

        block_starts = list(range(0, n_weeks - self.block_size + 1, self.block_size))
        if not block_starts:
            block_starts = [0]
        n_blocks = len(block_starts)

        # Sample n_blocks of the block start indices with replacement.
        # Namely, take the last example, therefore:
        #   sampled_indices (with replacement) = [2, 0, 2]
        #
        #   idx=2 → weeks [9, 10]  (short tail block)
        #   idx=0 → weeks [1, 2, 3, 4]
        #   idx=2 → weeks [9, 10]  (repeated)
        sampled_indices = self.rng.choice(
            len(block_starts), size=n_blocks, replace=True
        )

        # Concatenate the blocks into a single dataframe.
        pieces = []
        for idx in sampled_indices:
            start = block_starts[idx] # Get the start index of the block.
            sampled_block_weeks = train_weeks[start : start + self.block_size] # Get the weeks of the block.
            block_df = df[df[self.week_col].isin(sampled_block_weeks)].copy() # Get the dataframe of the block.
            pieces.append(block_df)

        return pd.concat(pieces, ignore_index=True) # Concatenate the blocks into a single dataframe.