from contextlib import contextmanager
import torch.nn as nn

_DROPOUT_TYPES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    nn.AlphaDropout, nn.FeatureAlphaDropout,
)

@contextmanager
def eval_dropouts(module: nn.Module):
    """
    Temporarily sets all Dropout submodules to eval (p=0)
    WITHOUT touching module.training. Gradients are unaffected.
    """
    layers = [m for m in module.modules() if isinstance(m, _DROPOUT_TYPES)]
    states  = [m.training for m in layers]
    for m in layers:
        m.eval()
    try:
        yield
    finally:
        for m, s in zip(layers, states):
            m.train(s)