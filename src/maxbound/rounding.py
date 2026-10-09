"""Floating-point helpers that keep our own float64 arithmetic sound.

Every bound in this package is computed in float64. Float64 rounds each result
to the nearest representable number, so a computed upper bound can land a hair
below the true upper bound. Three tools prevent that:

* ``up(x)`` / ``down(x)`` step to the next float64 above / below. One rounded
  operation is off by at most half a step, so one step outward is enough.
* ``inflate(x, n)`` turns a computed sum of at most n nonnegative products into
  a guaranteed upper bound on the exact sum. It uses Higham's constant
  gamma_n = n*u / (1 - n*u) (N. J. Higham, Accuracy and Stability of Numerical
  Algorithms, 2nd ed., SIAM 2002, Section 3.1). That bound holds for any order
  of summation and with fused multiply-add. ``deflate`` is the lower-bound twin.
* ``check_range`` stops the analysis if a value is non-finite, or so small that
  a product could underflow, which is the one case the rules above do not cover.

Exact zeros are left untouched by ``up`` and ``down``: a rounded sum is zero
only when the exact sum is zero, and (without underflow) a rounded product is
zero only when a factor is zero. This is what lets two identical models get a
bound of exactly 0.
"""
from __future__ import annotations

import math

import torch

from .errors import NumericalInconsistency

#: Unit roundoff (half the gap between 1.0 and the next float) of each format.
U64 = 2.0 ** -53
U32 = 2.0 ** -24

#: Nonzero magnitudes we accept inside the analysis. A product of two such
#: numbers stays far above the float64 underflow threshold (2**-1022).
_TINY_OK = 2.0 ** -400
_HUGE_OK = 2.0 ** 400


def unit_roundoff(dtype: torch.dtype) -> float:
    """Unit roundoff of the floating-point format ``dtype``."""
    if dtype == torch.float64:
        return U64
    if dtype == torch.float32:
        return U32
    raise ValueError(f"no rounding model for dtype {dtype}; use float32 or float64")


def gamma(n: int, u: float) -> float:
    """Higham's gamma_n = n*u / (1 - n*u), rounded up.

    A dot product of length n computed in a format with unit roundoff u is off by
    at most gamma_n * sum(|a_i * b_i|), whatever the summation order."""
    nu = n * u
    if nu >= 0.5:
        raise NumericalInconsistency(f"gamma({n}) is not usable: n*u = {nu:.3g} >= 0.5")
    return math.nextafter(nu / (1.0 - nu), math.inf)


def up(x: torch.Tensor) -> torch.Tensor:
    """The next float64 above x. Exact zeros stay zero (see module docstring)."""
    return torch.where(x == 0, x, torch.nextafter(x, torch.full_like(x, math.inf)))


def down(x: torch.Tensor) -> torch.Tensor:
    """The next float64 below x. Exact zeros stay zero."""
    return torch.where(x == 0, x, torch.nextafter(x, torch.full_like(x, -math.inf)))


def inflate(x: torch.Tensor, n_terms: int) -> torch.Tensor:
    """Guaranteed upper bound on a nonnegative quantity that was computed in
    float64 as a sum of at most ``n_terms`` products of nonnegative numbers.

    The computed value is at least (1 - gamma) times the exact one, so
    multiplying by (1 + 2*gamma) and stepping up one float covers it."""
    g = gamma(n_terms + 2, U64)
    return up(x * (1.0 + 2.0 * g))


def deflate(x: torch.Tensor, n_terms: int) -> torch.Tensor:
    """Guaranteed lower bound on a nonnegative quantity computed like in
    ``inflate`` (the computed value is at most (1 + gamma) times the exact one)."""
    g = gamma(n_terms + 2, U64)
    return torch.clamp(down(x * (1.0 - 2.0 * g)), min=0.0)


def check_range(*tensors: torch.Tensor, what: str = "value") -> None:
    """Fail loudly if a tensor holds a non-finite value, or a nonzero value so
    small or so large that the rounding rules above would not apply."""
    for t in tensors:
        if t.numel() == 0:
            continue
        a = t.abs()
        if not bool(torch.isfinite(a).all()):
            raise NumericalInconsistency(f"{what}: non-finite value in the analysis")
        nonzero = a[a != 0]
        if nonzero.numel() and (nonzero.min() < _TINY_OK or nonzero.max() > _HUGE_OK):
            raise NumericalInconsistency(
                f"{what}: magnitude outside [2^-400, 2^400], the rounding model does not apply"
            )
