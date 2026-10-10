"""maxbound: sound upper bounds on the output difference between a neural
network and its compiled version over a region of inputs.

    from maxbound import MaxBound, LinfBall
    from maxbound.compilers import quantize_weights

    cl_model = quantize_weights(model, bits=8)
    bound = MaxBound(model, cl_model, LinfBall(x0, eps=0.01, clip=(0.0, 1.0)))
    print(float(bound), bound.summary())
"""
from . import compilers, empirical
from .api import ASSUMPTIONS, Bound, MaxBound
from .config import MaxBoundConfig
from .domain import Box, LinfBall, as_box, domain_from_config
from .errors import NumericalInconsistency, UnsupportedModelError

__version__ = "0.1.0"

__all__ = [
    "MaxBound", "Bound", "ASSUMPTIONS", "MaxBoundConfig",
    "Box", "LinfBall", "as_box", "domain_from_config",
    "UnsupportedModelError", "NumericalInconsistency",
    "compilers", "empirical",
]
