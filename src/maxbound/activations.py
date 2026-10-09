"""What each activation function does to ranges of values.

Supported activations are all nondecreasing: relu, leaky_relu (slope in
[0, 1]), tanh and sigmoid. For each one this module provides

* ``sigma_bounds``: lower and upper enclosures of sigma(x) at float64 points;
* ``value_range``: the range of sigma over an interval;
* ``deriv_range``: bounds on sigma' over an interval (tanh and sigmoid);
* ``relaxation``: a line lam*z + mid with a band +-rad that contains sigma(z)
  for every z in an interval (used for the original activations h);
* ``diff_range``: the range of sigma(z + delta) - sigma(z) over a box of
  (z, delta) values (used for the difference d between the two networks).

The float64 functions torch.tanh, torch.sigmoid and torch.exp are assumed
accurate to a relative error of 2**-45 (their real error is near 2**-52; a test
checks the assumption against 50-digit decimal arithmetic).
"""
from __future__ import annotations

from typing import Tuple

import torch

from .intervals import Interval, hull, intersect, mul
from .rounding import down, up

#: Assumed relative accuracy of torch's float64 tanh, sigmoid and exp.
LIBM_REL = 2.0 ** -45
_SMALLEST = 2.0 ** -1074          # smallest positive float64
DERIV_MAX = {"tanh": 1.0, "sigmoid": 0.25}


def libm_enclosure(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn the result of a float64 library function into a guaranteed
    enclosure of the exact value, using the LIBM_REL assumption."""
    slack = v.abs() * LIBM_REL + _SMALLEST
    return down(v - slack), up(v + slack)


def sigma_bounds(kind: str, slope: float, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """lo <= sigma(x) <= hi at the float64 points x."""
    if kind == "relu":
        v = torch.clamp(x, min=0.0)
        return v, v
    if kind == "leaky_relu":
        neg = slope * x                       # one rounding; down/up enclose the exact product
        return torch.where(x >= 0, x, down(neg)), torch.where(x >= 0, x, up(neg))
    if kind == "tanh":
        lo, hi = libm_enclosure(torch.tanh(x))
        return lo.clamp(-1.0, 1.0), hi.clamp(-1.0, 1.0)
    if kind == "sigmoid":
        lo, hi = libm_enclosure(torch.sigmoid(x))
        return lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)
    raise ValueError(f"unknown activation {kind}")


def value_range(kind: str, slope: float, iv: Interval) -> Interval:
    """sigma is nondecreasing, so its range over [lo, hi] is [sigma(lo), sigma(hi)]."""
    return Interval(sigma_bounds(kind, slope, iv.lo)[0], sigma_bounds(kind, slope, iv.hi)[1])


#: Computed slopes below 2 * _SLOPE_FLOOR are treated as "somewhere in
#: [0, 4 * _SLOPE_FLOOR]", which keeps every value inside the range where
#: check_range accepts it.
_SLOPE_FLOOR = 2.0 ** -300


def _deriv(kind: str, x: torch.Tensor) -> torch.Tensor:
    """sigma'(x) computed only from exp (covered by assumption A2), in forms
    that cannot overflow: tanh'(x) = 4 t / (1 + t)^2 with t = exp(-2|x|), and
    sigmoid'(x) = t / (1 + t)^2 with t = exp(-|x|)."""
    if kind == "tanh":
        t = torch.exp(-2.0 * x.abs())
        return 4.0 * t / (1.0 + t) ** 2
    t = torch.exp(-x.abs())
    return t / (1.0 + t) ** 2


def deriv_range(kind: str, iv: Interval) -> Interval:
    """Bounds on sigma' over iv, for tanh and sigmoid. Both derivatives are even
    and decrease as |x| grows, so the largest value is at the point of iv closest
    to 0 and the smallest is at one of the two ends.

    The computed values have a relative error of a few times LIBM_REL, covered by
    the 2^-40 margin. A computed slope below 2^-299 (where exp may have
    underflowed) is replaced by the enclosure [0, 2^-298]: a lower end of 0, and an
    upper end of 2^-298 when the computed largest slope is below 2^-299."""
    if kind not in DERIV_MAX:
        raise ValueError(f"deriv_range is for smooth activations, not {kind}")
    x0 = torch.minimum(torch.maximum(torch.zeros_like(iv.lo), iv.lo), iv.hi)
    top = _deriv(kind, x0)
    bottom = torch.minimum(_deriv(kind, iv.lo), _deriv(kind, iv.hi))
    lo = torch.where(bottom < 2 * _SLOPE_FLOOR, 0.0, down(bottom * (1.0 - 2.0 ** -40)))
    hi = torch.where(top < 2 * _SLOPE_FLOOR, _SLOPE_FLOOR * 4, up(top * (1.0 + 2.0 ** -40)))
    return Interval(lo, torch.clamp(hi, max=DERIV_MAX[kind]))


def relaxation(kind: str, slope: float, iv: Interval) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(lam, mid, rad) with |sigma(z) - lam*z - mid| <= rad for all z in iv.

    relu / leaky_relu (meant for l < 0 < u): lam is the slope of the chord. For
    ANY lam in [slope, 1], g(z) = sigma(z) - lam*z is >= 0 and largest at an end
    of [l, u], so g ranges over [0, max(g(l), g(u))]. Soundness therefore does
    not depend on lam being computed exactly (DeepZ's ReLU transformer).

    tanh / sigmoid: lam is a lower bound on sigma' over iv, so g is
    nondecreasing on iv and ranges over [g(l), g(u)]."""
    l, u = iv.lo, iv.hi
    if kind in ("relu", "leaky_relu"):
        a = slope if kind == "leaky_relu" else 0.0
        width = torch.where(u > l, u - l, torch.ones_like(u))
        lam = torch.clamp((u - a * l) / width, min=a, max=1.0)
        g_l, g_u = (a - lam) * l, (1.0 - lam) * u
        gmax = up(torch.clamp(torch.maximum(g_l, g_u), min=0.0) * (1.0 + 2.0 ** -40))
        half = 0.5 * gmax
        return lam, half, half
    lam = deriv_range(kind, iv).lo
    g_lo = down(sigma_bounds(kind, slope, l)[0] - up(lam * l))
    g_hi = up(sigma_bounds(kind, slope, u)[1] - down(lam * u))
    g = Interval(g_lo, torch.maximum(g_lo, g_hi))
    return lam, g.center(), g.radius()


def diff_range(kind: str, slope: float, z: Interval, delta: Interval, zp: Interval) -> Interval:
    """Range of f(z, delta) = sigma(z + delta) - sigma(z) for z in ``z`` and
    delta in ``delta``; ``zp`` encloses z + delta (the compiled pre-activation).

    relu / leaky_relu: f is nondecreasing in delta (sigma is nondecreasing), and
    for fixed delta it is monotone in z (sigma is convex), so its extremes over
    the box sit at corners: min at delta = delta.lo, max at delta = delta.hi.

    tanh / sigmoid: by the mean value theorem f = sigma'(xi) * delta for some xi
    between z and z + delta, which lies in hull(z, zp).

    Both are intersected with the independent enclosure
    [sigma(zp.lo) - sigma(z.hi), sigma(zp.hi) - sigma(z.lo)]. If delta is exactly
    zero, f is exactly zero."""
    if kind in ("relu", "leaky_relu"):
        def f_lo(zz: torch.Tensor, dd: torch.Tensor) -> torch.Tensor:
            """Guaranteed lower value of sigma(zz + dd) - sigma(zz) at float64 points."""
            return down(sigma_bounds(kind, slope, down(zz + dd))[0] - sigma_bounds(kind, slope, zz)[1])

        def f_hi(zz: torch.Tensor, dd: torch.Tensor) -> torch.Tensor:
            """Guaranteed upper value of sigma(zz + dd) - sigma(zz) at float64 points."""
            return up(sigma_bounds(kind, slope, up(zz + dd))[1] - sigma_bounds(kind, slope, zz)[0])

        box = Interval(torch.minimum(f_lo(z.lo, delta.lo), f_lo(z.hi, delta.lo)),
                       torch.maximum(f_hi(z.lo, delta.hi), f_hi(z.hi, delta.hi)))
    else:
        box = mul(deriv_range(kind, hull(z, zp)), delta)
    independent = Interval(down(sigma_bounds(kind, slope, zp.lo)[0] - sigma_bounds(kind, slope, z.hi)[1]),
                           up(sigma_bounds(kind, slope, zp.hi)[1] - sigma_bounds(kind, slope, z.lo)[0]))
    out = intersect(box, independent)
    zero = (delta.lo == 0) & (delta.hi == 0)
    return Interval(torch.where(zero, 0.0, out.lo), torch.where(zero, 0.0, out.hi))
