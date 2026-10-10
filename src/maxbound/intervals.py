"""Interval arithmetic in float64 with outward rounding.

An Interval is a pair of tensors ``(lo, hi)`` that is guaranteed to contain the
true value of some quantity, elementwise, for every input in the input region.
Every operation rounds its result outward (``down`` for lower ends, ``up`` for
upper ends, see rounding.py), so the guarantee survives float64 arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from .errors import NumericalInconsistency
from .rounding import down, up


@dataclass
class Interval:
    """Elementwise interval ``[lo, hi]`` stored as two float64 tensors."""

    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.lo.dtype != torch.float64 or self.hi.dtype != torch.float64:
            raise TypeError("Interval endpoints must be float64 tensors")
        if self.lo.shape != self.hi.shape:
            raise ValueError(f"shape mismatch: {tuple(self.lo.shape)} vs {tuple(self.hi.shape)}")
        if bool(torch.isnan(self.lo).any() or torch.isnan(self.hi).any()) or bool((self.lo > self.hi).any()):
            raise NumericalInconsistency("interval with NaN or with lower end above upper end")

    @staticmethod
    def point(x: torch.Tensor) -> "Interval":
        """A zero-width interval at x."""
        x = x.detach().to(torch.float64)
        return Interval(x.clone(), x.clone())

    @property
    def shape(self) -> torch.Size:
        """Shape of the interval."""
        return self.lo.shape

    def center(self) -> torch.Tensor:
        """Any point works as a center; ``radius`` is computed to be safe for it."""
        return 0.5 * (self.lo + self.hi)

    def radius(self) -> torch.Tensor:
        """A radius r such that [center - r, center + r] contains [lo, hi]."""
        c = self.center()
        return up(torch.maximum(self.hi - c, c - self.lo))

    def mag(self) -> torch.Tensor:
        """Largest absolute value any point of the interval can have."""
        return torch.maximum(self.lo.abs(), self.hi.abs())

    def reshape(self, *shape) -> "Interval":
        """Reshape both endpoints."""
        return Interval(self.lo.reshape(*shape), self.hi.reshape(*shape))

    def contains(self, x: torch.Tensor) -> bool:
        """True if x lies inside the interval."""
        x = x.to(torch.float64)
        return bool(((x >= self.lo) & (x <= self.hi)).all())


Scalar = Union[float, torch.Tensor]


def add(a: Interval, b: Interval) -> Interval:
    """All sums x + y with x in a and y in b."""
    return Interval(down(a.lo + b.lo), up(a.hi + b.hi))


def sub(a: Interval, b: Interval) -> Interval:
    """All values x - y with x in a and y in b."""
    return Interval(down(a.lo - b.hi), up(a.hi - b.lo))


def scale(a: Interval, k: Scalar) -> Interval:
    """All values k * x with x in a, for an exact constant (or tensor) k."""
    k = torch.as_tensor(k, dtype=torch.float64)
    p, q = a.lo * k, a.hi * k
    return Interval(down(torch.minimum(p, q)), up(torch.maximum(p, q)))


def mul(a: Interval, b: Interval) -> Interval:
    """All products x * y with x in a and y in b. A product is bilinear, so its
    extremes over a box sit at the four corners."""
    corners = torch.stack([a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi])
    return Interval(down(corners.min(0).values), up(corners.max(0).values))


def hull(a: Interval, b: Interval) -> Interval:
    """Smallest interval containing both."""
    return Interval(torch.minimum(a.lo, b.lo), torch.maximum(a.hi, b.hi))


def intersect(a: Interval, b: Interval) -> Interval:
    """Intersection of two sound enclosures of the same quantity. The true value
    lies in both, so an empty intersection can only mean a bug: we raise."""
    lo, hi = torch.maximum(a.lo, b.lo), torch.minimum(a.hi, b.hi)
    if bool((lo > hi).any()):
        gap = float((lo - hi).max())
        raise NumericalInconsistency(f"two sound enclosures do not overlap (gap {gap:.3e}); this is a bug")
    return Interval(lo, hi)
