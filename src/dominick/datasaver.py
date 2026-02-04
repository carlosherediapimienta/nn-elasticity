import pandas as pd
from pathlib import Path


class DominickDataSaver:
    def __init__(self):
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    def save(self, filename: str, df: pd.DataFrame) -> None:
        df.to_csv(self.data_dir / filename, index=False)