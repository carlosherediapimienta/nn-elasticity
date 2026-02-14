class MetricAggregator:
    """
    Aggregate metrics from multiple batches.
    Public API: add(), compute().
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the aggregator."""
        self.agg = {}
        self.n = 0
    
    def add(self, logs: dict, batch_size: int):
        """
        Add metrics from a batch.
        
        Args:
            logs: dict with metrics
            batch_size: size of the batch
        """
        for k, v in logs.items():
            self.agg[k] = self.agg.get(k, 0.0) + float(v) * batch_size
        self.n += batch_size
    
    def compute(self) -> dict[str, float]:
        """
        Calculate weighted averages.
        
        Returns:
            dict with aggregated metrics
        """
        result = {}
        for k in self.agg:
            result[k] = self.agg[k] / max(self.n, 1)
        return result