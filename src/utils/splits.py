import numpy as np
import pandas as pd
from typing import List, Tuple


class TemporalSplitter:
    """Responsabilidad única: particionar datos temporales en train/val."""

    def __init__(self, week_col: str = "week_id") -> None:
        self.week_col = week_col

    def single_split(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Un solo split cronológico: primeros train_frac% de semanas = train, resto = val.
        """
        sorted_weeks = sorted(df[self.week_col].unique())
        threshold_idx = int(len(sorted_weeks) * train_frac)
        week_threshold = sorted_weeks[threshold_idx]

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
        Expanding window: cada fold entrena solo con pasado y valida en futuro.
        Mide generalización temporal real (sin leakage).

        Args:
            df: DataFrame con columna week_col.
            n_folds: Número de folds.
            min_train_frac: Fracción mínima de semanas para el primer train (ej. 0.5 = 50%).

        Returns:
            Lista de (train_df, val_df) de longitud <= n_folds.
        """
        sorted_weeks = sorted(df[self.week_col].unique())
        n_weeks = len(sorted_weeks)

        min_train_weeks = max(1, int(n_weeks * min_train_frac))
        remaining_weeks = n_weeks - min_train_weeks
        val_size = max(1, remaining_weeks // n_folds)

        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for i in range(n_folds):
            train_end = min_train_weeks + i * val_size
            val_end = min(train_end + val_size, n_weeks)

            if val_end <= train_end:
                break

            train_weeks = sorted_weeks[:train_end]
            val_weeks = sorted_weeks[train_end:val_end]

            train_df = df[df[self.week_col].isin(train_weeks)].copy()
            val_df = df[df[self.week_col].isin(val_weeks)].copy()

            if len(val_df) == 0:
                continue

            folds.append((train_df, val_df))

        return folds


class BlockBootstrapSampler:
    """Responsabilidad única: re-muestreo por bloques de semanas para bootstrap."""

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
        Block bootstrap con reemplazo: conserva multiplicidad de bloques repetidos.
        """
        train_weeks = (
            sorted(train_weeks)
            if not isinstance(train_weeks, np.ndarray)
            else np.sort(train_weeks).tolist()
        )
        n_weeks = len(train_weeks)

        if n_weeks < self.block_size:
            return df[df[self.week_col].isin(train_weeks)].copy()

        block_starts = list(range(0, n_weeks - self.block_size + 1, self.block_size))
        if not block_starts:
            block_starts = [0]
        n_blocks = len(block_starts)

        sampled_indices = self.rng.choice(
            len(block_starts), size=n_blocks, replace=True
        )

        pieces = []
        for idx in sampled_indices:
            start = block_starts[idx]
            sampled_block_weeks = train_weeks[start : start + self.block_size]
            block_df = df[df[self.week_col].isin(sampled_block_weeks)].copy()
            pieces.append(block_df)

        return pd.concat(pieces, ignore_index=True)