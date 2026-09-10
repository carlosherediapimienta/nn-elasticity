from .preprocessing import ColumnEncoder
from .dataset import DataLoaderFactory
from .spline import StatisticsCalculator, KnotGenerator
from .spline.spline_builder import SplineBuilder

# Backward compatibility
def factorize_col(df, col):
    """Wrapper for compatibility with legacy code."""
    encoder = ColumnEncoder()
    return encoder.factorize(df, col)

def build_price_spline_from_train(x_train_np, K=16, basis_type="truncated_cubic"):
    """Wrapper for compatibility with legacy code."""
    builder = SplineBuilder()
    result = builder.build_from_data(x_train_np, n_basis=K, basis_type=basis_type)
    return result["knots"], result["mean"], result["std"]

__all__ = [
    # Classes
    'ColumnEncoder',
    'DataLoaderFactory',
    'StatisticsCalculator',
    'KnotGenerator',
    'SplineBuilder',
    # Legacy functions
    'factorize_col',
    'build_price_spline_from_train',
]