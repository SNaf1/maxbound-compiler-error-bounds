"""Bounds for a final softmax layer.

Let p = softmax(z). Its Jacobian is J[i, j] = p_i * (1[i == j] - p_j), so

* row i has absolute sum 2 * p_i * (1 - p_i), which is at most 1/2;
* every row sums to zero, so J (delta - c * 1) = J delta for any constant c:
  adding the same number to all logits does not change softmax.

By the mean value theorem along the segment from z to z' = z + delta,

    |softmax_i(z') - softmax_i(z)| <= max 2 p_i (1 - p_i) * min_c ||delta - c||_inf
                                   =  max p_i (1 - p_i) * (max_j delta_j - min_j delta_j)

where the max runs over the probabilities p_i met along the segment. We bound
max_j delta_j - min_j delta_j with the pairwise forms of the difference, and the
probabilities with the pairwise forms of both networks' logits. For a confident
classifier, p_i (1 - p_i) is far below 1/4, which makes this much tighter than
the global constant 1/2.
"""
from __future__ import annotations

from typing import List

import torch

from .activations import libm_enclosure
from .intervals import Interval
from .rounding import U64, deflate, down, inflate, up
from .zonotope import Zonotope

_EXP_CAP = 700.0     # exp(700) is finite in float64; larger exponents are capped safely


def pair_bounds(z: Zonotope) -> Interval:
    """Entry [i, j] encloses value_i - value_j over the input region. It is
    computed from the forms, so shared terms cancel before we take bounds."""
    c, e = z.c.flatten(), z.e.flatten()
    G = z.G.reshape(z.rows, c.numel())
    pc = c[:, None] - c[None, :]
    pG = G[:, :, None] - G[:, None, :]
    pe = inflate(e[:, None] + e[None, :] + 2 * U64 * (pc.abs() + pG.abs().sum(0)), z.rows + 4)
    out = Zonotope(pc, pG, pe).bounds()
    eye = torch.eye(c.numel(), dtype=torch.bool)
    return Interval(torch.where(eye, 0.0, out.lo), torch.where(eye, 0.0, out.hi))


def prob_range(pairs: List[Interval]) -> Interval:
    """Range of every probability p_i = 1 / (1 + sum_{j != i} exp(v_j - v_i)),
    where each pairs[k][j, i] encloses v_j - v_i for one network. Taking the hull
    over the networks covers every point on the segment between them."""
    lo = torch.stack([p.lo for p in pairs]).min(0).values
    hi = torch.stack([p.hi for p in pairs]).max(0).values
    n = lo.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    e_hi = libm_enclosure(torch.exp(hi.clamp(max=_EXP_CAP)))[1]
    e_lo = libm_enclosure(torch.exp(lo.clamp(max=_EXP_CAP)))[0].clamp(min=0.0)
    s_hi = inflate((e_hi * off).sum(0), n)
    s_lo = deflate((e_lo * off).sum(0), n)
    p_lo = down(1.0 / up(1.0 + s_hi))
    p_lo = torch.where(((hi > _EXP_CAP) & off).any(0), torch.zeros_like(p_lo), p_lo)
    p_hi = torch.clamp(up(1.0 / down(1.0 + s_lo)), max=1.0)
    return Interval(p_lo.clamp(min=0.0), p_hi)


def max_pq(p: Interval) -> torch.Tensor:
    """Upper bound on max p(1 - p) over each interval of probabilities.
    p(1 - p) rises up to p = 1/2 and falls after it."""
    def pq(x: torch.Tensor) -> torch.Tensor:
        """Upper bound on x (1 - x)."""
        return up(x * up(1.0 - x))
    inside = (p.lo <= 0.5) & (p.hi >= 0.5)
    return torch.where(inside, torch.full_like(p.lo, 0.25), torch.where(p.hi < 0.5, pq(p.hi), pq(p.lo)))


def difference_bound(diff: Zonotope, a: Zonotope, b: Zonotope) -> torch.Tensor:
    """Per output i: bound on |softmax_i(b) - softmax_i(a)| over the input region,
    given the forms of both networks' logits (a, b) and of their difference."""
    spread = pair_bounds(diff).hi.clamp(min=0.0).max()
    p = prob_range([pair_bounds(a), pair_bounds(b)])
    return up(max_pq(p) * spread)
