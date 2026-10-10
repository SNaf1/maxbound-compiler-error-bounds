"""Affine forms (zonotopes) whose noise symbols are shared by both networks.

Each neuron's value is written as

    value = c + G[0] * eps_0 + G[1] * eps_1 + ... + xi

where every noise symbol eps_k ranges over [-1, 1] and |xi| <= e. The first
symbols stand for the input coordinates (x_i = center_i + radius_i * eps_i).
New symbols are added whenever a nonlinear step cannot be written exactly.

Why this matters: the original and the compiled network run on the SAME input,
so their forms use the SAME input symbols. Subtracting two forms cancels the
shared part exactly. Intervals cannot do that, because they forget which input
caused which value. This is the zonotope domain of DeepZ (Singh et al.,
"Fast and Effective Robustness Certification", NeurIPS 2018), used here for two
networks at once.

The guarantee every function below keeps: for every input x in X there is ONE
assignment of all symbols in [-1, 1], shared by every form, and for each neuron
some |xi| <= e, such that every form equals the true value of its neuron.

Storage: c has the layer's shape S, G has shape (rows, *S), e has shape S.
G may have fewer rows than there are symbols; missing rows are zero. ``e``
absorbs our own float64 rounding errors and any part kept as a plain interval.
When ``pool`` is None we never create symbols, and every form is just
c +- e: that is plain interval arithmetic, the "interval" analysis mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from .intervals import Interval
from .rounding import U64, check_range, down, gamma, inflate, up


class SymbolPool:
    """Hands out indices for new noise symbols so no index is reused."""

    def __init__(self) -> None:
        self.count = 0

    def allocate(self, k: int) -> int:
        """Reserve k new symbols and return the index of the first."""
        start = self.count
        self.count += int(k)
        return start


@dataclass
class Zonotope:
    c: torch.Tensor
    G: torch.Tensor
    e: torch.Tensor

    @property
    def shape(self) -> torch.Size:
        """Shape of the layer (one form per neuron)."""
        return self.c.shape

    @property
    def rows(self) -> int:
        """Number of generator rows stored (missing rows are zero)."""
        return int(self.G.shape[0])

    @staticmethod
    def zeros(shape) -> "Zonotope":
        """The form that is exactly 0 for every neuron."""
        z = torch.zeros(shape, dtype=torch.float64)
        return Zonotope(z, z.new_zeros((0, *z.shape)), z.clone())

    @staticmethod
    def from_box(box: Interval, pool: Optional[SymbolPool]) -> "Zonotope":
        """x_i = center_i + radius_i * eps_i: one input symbol per coordinate
        that is not fixed."""
        r = box.radius()
        return fresh(box.center(), r, r > 0, pool)

    def radius(self) -> torch.Tensor:
        """Upper bound on how far a form can move from its center c."""
        return inflate(self.G.abs().sum(0) + self.e, self.rows + 2)

    def magnitude(self) -> torch.Tensor:
        """Upper bound on |value| for every input in X."""
        return inflate(self.c.abs() + self.G.abs().sum(0) + self.e, self.rows + 3)

    def bounds(self) -> Interval:
        """Concretize: the interval of values the form can take. Exact for the
        form, because each symbol moves independently over [-1, 1]."""
        r = self.radius()
        return Interval(down(self.c - r), up(self.c + r))

    def padded(self, rows: int) -> "Zonotope":
        """The same form with zero rows appended up to ``rows``."""
        if rows == self.rows:
            return self
        extra = self.G.new_zeros((rows - self.rows, *self.shape))
        return Zonotope(self.c, torch.cat([self.G, extra]), self.e)

    def reshape(self, shape) -> "Zonotope":
        """Reshape the forms to a new layer shape."""
        return Zonotope(self.c.reshape(shape), self.G.reshape(self.rows, *shape), self.e.reshape(shape))


def _checked(z: Zonotope, what: str) -> Zonotope:
    check_range(z.c, z.G, z.e, what=what)
    return z


def fresh(center: torch.Tensor, radius: torch.Tensor, mask: torch.Tensor,
          pool: Optional[SymbolPool]) -> Zonotope:
    """The form center + radius * (a new symbol) for each neuron in ``mask``,
    and just center elsewhere. Without a pool, the radius goes into e."""
    center = center.clone()
    radius = torch.where(mask, radius, torch.zeros_like(radius))
    if pool is None:
        return _checked(Zonotope(center, center.new_zeros((0, *center.shape)), radius), "box")
    idx = mask.flatten().nonzero().squeeze(1)
    k = idx.numel()
    start = pool.allocate(k)
    G = center.new_zeros((start + k, center.numel()))
    G[start + torch.arange(k), idx] = radius.flatten()[idx]
    return _checked(Zonotope(center, G.view(start + k, *center.shape), torch.zeros_like(center)), "fresh")


def affine(z: Zonotope, op, bias: bool = True) -> Zonotope:
    """Apply y = W x + b to a form. Exact in real arithmetic: the center and every
    generator row go through W. What remains is rounding:

    * c and the rows of G are computed in float64, each output off by at most
      gamma_{n+1} * (|W| |c| + |b|) and gamma_n * |W| |G_k| (Higham);
    * the old radius e goes through |W| (an interval through a linear map);
    * if W or b carry an uncertainty radius, add |dW| |x| + |db|.
    """
    n = op.fan_in()
    c = op.linear(z.c.unsqueeze(0))[0]
    if bias:
        c = c + op.bias_full()
    G = op.linear(z.G) if z.rows else c.new_zeros((0, *c.shape))
    abs_G = z.G.abs().sum(0)
    abs_c, abs_Gw, abs_e = op.abs_linear(torch.stack([z.c.abs(), abs_G, z.e]))
    terms = abs_c + abs_Gw
    if bias:
        terms = terms + op.bias_full().abs()
    err = abs_e + gamma(n + 1, U64) * terms
    if op.weight_err is not None:
        err = err + op.err_linear((z.c.abs() + abs_G + z.e).unsqueeze(0))[0]
    if bias and op.bias_err is not None:
        err = err + op.per_output(op.bias_err)
    return _checked(Zonotope(c, G, inflate(err, n + z.rows + 8)), op.name or op.kind)


def _combine(a: Zonotope, b: Zonotope, sign: float) -> Zonotope:
    rows = max(a.rows, b.rows)
    a, b = a.padded(rows), b.padded(rows)
    c, G = a.c + sign * b.c, a.G + sign * b.G
    # x + y is exact when x or y is zero; otherwise its error is at most 2u|result|
    both_c = (a.c != 0) & (b.c != 0)
    both_G = (a.G != 0) & (b.G != 0)
    slack = 2 * U64 * (c.abs() * both_c + (G.abs() * both_G).sum(0))
    return _checked(Zonotope(c, G, inflate(a.e + b.e + slack, rows + 4)), "sum")


def add(a: Zonotope, b: Zonotope) -> Zonotope:
    """Form of a + b (coefficients of shared symbols add up)."""
    return _combine(a, b, 1.0)


def sub(a: Zonotope, b: Zonotope) -> Zonotope:
    """Form of a - b (coefficients of shared symbols cancel)."""
    return _combine(a, b, -1.0)


def scale(a: Zonotope, k: Union[float, torch.Tensor]) -> Zonotope:
    """Multiply each neuron's form by its own exact constant k."""
    k = torch.as_tensor(k, dtype=torch.float64).expand(a.shape)
    c, G = a.c * k, a.G * k
    slack = 2 * U64 * (c.abs() + G.abs().sum(0))
    return _checked(Zonotope(c, G, inflate(a.e * k.abs() + slack, a.rows + 4)), "scale")


def shift(a: Zonotope, t: torch.Tensor) -> Zonotope:
    """Add an exact constant t to each neuron's form."""
    c = a.c + t
    slack = 2 * U64 * c.abs() * ((a.c != 0) & (t != 0))
    return _checked(Zonotope(c, a.G, inflate(a.e + slack, 3)), "shift")


def select(mask: torch.Tensor, a: Zonotope, b: Zonotope) -> Zonotope:
    """Per neuron: a's form where mask is true, b's form elsewhere."""
    rows = max(a.rows, b.rows)
    a, b = a.padded(rows), b.padded(rows)
    return Zonotope(torch.where(mask, a.c, b.c), torch.where(mask, a.G, b.G), torch.where(mask, a.e, b.e))
