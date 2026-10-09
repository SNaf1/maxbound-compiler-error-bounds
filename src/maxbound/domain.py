"""Input domains X: the set of inputs the bound must cover.

A domain describes the inputs of ONE example, without a batch dimension, for
example a 784-vector for an MNIST MLP or a (1, 28, 28) image for a CNN.

* ``Box(lower, upper)``: every x with lower <= x <= upper in every coordinate.
* ``LinfBall(center, eps, clip=None)``: every x with |x - center| <= eps in
  every coordinate, optionally intersected with [clip[0], clip[1]] (for example
  the valid pixel range).

Both become a Box before the analysis. The conversion rounds outward, so the
Box always contains the domain the user described. Domains can also be written
as a dict (``domain_from_config``), which is how JSON configs describe them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Union

import torch

from .intervals import Interval
from .rounding import down, up


def _f64(x: Any) -> torch.Tensor:
    """Convert to float64 directly. (torch.as_tensor on a Python list would first
    round to float32 and move the domain.)"""
    return torch.as_tensor(x, dtype=torch.float64).detach().clone()


@dataclass
class Box:
    """All x with ``lower <= x <= upper`` elementwise (float64 tensors)."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        self.lower, self.upper = _f64(self.lower), _f64(self.upper)
        if self.lower.shape != self.upper.shape:
            raise ValueError(f"lower {tuple(self.lower.shape)} and upper {tuple(self.upper.shape)} differ in shape")
        if not bool(torch.isfinite(self.lower).all() and torch.isfinite(self.upper).all()):
            raise ValueError("Box bounds must be finite")
        if bool((self.lower > self.upper).any()):
            raise ValueError("Box needs lower <= upper in every coordinate")

    @property
    def shape(self) -> torch.Size:
        """Shape of one input (no batch dimension)."""
        return self.lower.shape

    def interval(self) -> Interval:
        """The box as an Interval."""
        return Interval(self.lower.clone(), self.upper.clone())

    def float32_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Smallest and largest float32 values that lie inside the box, per
        coordinate. Models run in float32, so these are the inputs that exist."""
        inf = torch.full(self.shape, math.inf, dtype=torch.float32)
        lo = self.lower.to(torch.float32)
        lo = torch.where(lo.double() < self.lower, torch.nextafter(lo, inf), lo)
        hi = self.upper.to(torch.float32)
        hi = torch.where(hi.double() > self.upper, torch.nextafter(hi, -inf), hi)
        if bool((lo > hi).any()):
            raise ValueError("the box contains no float32 value in some coordinate")
        return lo, hi

    def center_float32(self) -> torch.Tensor:
        """A float32 point inside the box: its midpoint, rounded into the box."""
        lo, hi = self.float32_bounds()
        mid = (0.5 * (lo.double() + hi.double())).to(torch.float32)
        return torch.minimum(torch.maximum(mid, lo), hi)

    def sample(self, n: int, seed: int = 0, corner_fraction: float = 0.25) -> torch.Tensor:
        """``n`` float32 points inside the box, shape (n, *shape). A fraction of
        them are random corners, where differences are often largest."""
        lo, hi = self.float32_bounds()
        g = torch.Generator().manual_seed(seed)
        t = torch.rand((n, *self.shape), generator=g, dtype=torch.float64)
        x = (lo.double() + t * (hi.double() - lo.double())).to(torch.float32)
        k = int(n * corner_fraction)
        if k:
            pick = torch.rand((k, *self.shape), generator=g) < 0.5
            x[:k] = torch.where(pick, hi, lo)
        return torch.minimum(torch.maximum(x, lo), hi)

    def contains(self, x: torch.Tensor) -> bool:
        """True if x lies inside the box."""
        x = x.to(torch.float64)
        return bool(((x >= self.lower) & (x <= self.upper)).all())


@dataclass
class LinfBall:
    """All x with ``|x - center| <= eps`` in every coordinate, optionally clipped
    to ``[clip[0], clip[1]]``."""

    center: torch.Tensor
    eps: float
    clip: Optional[Tuple[float, float]] = None

    def to_box(self) -> Box:
        """The smallest box containing the ball (and the clip range), rounded outward."""
        c = _f64(self.center)
        if self.eps < 0:
            raise ValueError("eps must be nonnegative")
        lo, hi = down(c - self.eps), up(c + self.eps)
        if self.clip is not None:
            lo = torch.clamp(lo, min=float(self.clip[0]))
            hi = torch.clamp(hi, max=float(self.clip[1]))
            if bool((lo > hi).any()):
                raise ValueError("the clip range does not meet the ball")
        return Box(lo, hi)


Domain = Union[Box, LinfBall]


def domain_from_config(cfg: Mapping[str, Any]) -> Domain:
    """Build a domain from a dict, for example
    ``{"type": "box", "lower": [...], "upper": [...]}`` or
    ``{"type": "linf_ball", "center": [...], "eps": 0.1, "clip": [0, 1]}``."""
    kind = cfg.get("type")
    if kind == "box":
        return Box(_f64(cfg["lower"]), _f64(cfg["upper"]))
    if kind == "linf_ball":
        clip = cfg.get("clip")
        return LinfBall(_f64(cfg["center"]), float(cfg["eps"]), tuple(clip) if clip is not None else None)
    raise ValueError(f"unknown domain type {kind!r}; use 'box' or 'linf_ball'")


def as_box(X: Any) -> Box:
    """Accept a Box, a LinfBall, a config dict or a (lower, upper) pair."""
    if isinstance(X, Box):
        return X
    if isinstance(X, LinfBall):
        return X.to_box()
    if isinstance(X, Mapping):
        return as_box(domain_from_config(X))
    if isinstance(X, (tuple, list)) and len(X) == 2:
        return Box(X[0], X[1])
    raise TypeError("X must be a Box, a LinfBall, a dict config or a (lower, upper) pair")
