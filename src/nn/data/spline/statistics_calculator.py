import numpy as np

class StatisticsCalculator:
    """
    Computes the statistics of the data to normalize the spline.
    Public API: compute_stats().
    """
    def compute_stats(self, data: np.ndarray) -> dict[str, float]:
        """
        Computes mean and standard deviation of the data to normalize the spline.
        Filters out non-finite values (NaN, inf) before computing statistics.
        The std has a minimum of 0.2 to avoid division by zero
        for low-variability data (e.g. nearly constant prices).
        Args:
            data: array of price values (log_price) for a single product.
        Returns:
            dict with "mean" and "std" to use as shift/scale for the spline.
        Raises:
            ValueError: if no finite values are found in the data.
        """
        x = np.asarray(data, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0: # If there are no finite values, raise an error
            raise ValueError("There are no finite values for the spline statistics.")

        mean = float(x.mean()) # Mean of the data
        # We compute the standard deviation of the data over all the data points (ddof=0)
        # Besides, we add a minimum of 0.2 to avoid division by zero for low-variability data.
        # With this, the training process will not be unstable when the data is nearly constant.
        std  = float(max(x.std(ddof=0), 0.2))
        return {"mean": mean, "std": std} # Return the mean and standard deviation