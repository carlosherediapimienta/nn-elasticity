# statistics_calculator.py
import numpy as np

class StatisticsCalculator:
    def compute_stats(self, data: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
        x = np.asarray(data, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("There are no finite values for the spline statistics.")

        mean = float(x.mean())
        std  = float(max(x.std(ddof=0), 0.2) + eps)
        return {"mean": mean, "std": std}