import pandas as pd
from pathlib import Path


class DominickDataLoader:
    def __init__(self):
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    def load(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / filename, low_memory=False)