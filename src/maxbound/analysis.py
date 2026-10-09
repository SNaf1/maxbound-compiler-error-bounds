"""The analysis engine: one loop over the layers, four settings.

Settings (``METHODS``):
    "zonotope"           differential analysis with shared affine forms (default, tightest)
    "interval"           differential analysis with plain intervals (ReluDiff-style)
    "zonotope-separate"  each network on its own, forms share only the input symbols
    "interval-separate"  each network on its own with intervals (naive IBP)

Differential analysis follows two quantities through the network:
    h = the original network's values,  d = (compiled values) - (original values).

At an affine layer (z = W h + b in the original, z' = W' h' + b' compiled):

    delta = z' - z = W' d + (W' - W) h + (b' - b)

which is exact for forms; only rounding is added.

At an activation sigma, with z and delta the incoming values and z' = z + delta:
    h_new = sigma(z), via ``_act_value``;
    d_new = sigma(z + delta) - sigma(z), via ``_act_diff``.
For ReLU, d_new is exact whenever both networks have the neuron stably on or
stably off (or one stably on and the other stably off). Only neurons that can
switch inside X fall back to an enclosing interval. This is where differential
analysis wins: d stays as small as the weight change, while bounding the two
networks separately leaves the full uncertainty of each network in the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from . import zonotope as zt
from .activations import deriv_range, diff_range, relaxation, value_range
from .graph import ActOp, AffineOp, Op, ReshapeOp
from .intervals import Interval, hull
from .rounding import up
from .zonotope import SymbolPool, Zonotope

#: method name -> (representation, differential?)
METHODS = {
    "zonotope": ("zonotope", True),
    "interval": ("interval", True),
    "zonotope-separate": ("zonotope", False),
    "interval-separate": ("interval", False),
}


@dataclass
class Result:
    a: Zonotope        # final values of the original network (before any softmax)
    b: Zonotope        # final values of the compiled network
    diff: Zonotope     # b - a
    relaxed: int       # neurons where a nonlinear step had to be approximated


def run(ops_a: List[Op], ops_b: List[Op], box: Interval, method: str) -> Result:
    """Run one analysis over the input box. ``ops_a``/``ops_b`` must line up for
    the differential methods (see graph.alignment_problem)."""
    mode, differential = METHODS[method]
    pool = SymbolPool() if mode == "zonotope" else None
    x = Zonotope.from_box(box, pool)
    counter = [0]
    if differential:
        h, d = x, Zonotope.zeros(x.shape)
        for op_a, op_b in zip(ops_a, ops_b):
            h, d = _diff_step(op_a, op_b, h, d, pool, counter)
        return Result(h, zt.add(h, d), d, counter[0])
    a = _run_single(ops_a, x, pool, counter)
    b = _run_single(ops_b, x, pool, counter)
    return Result(a, b, zt.sub(b, a), counter[0])


def _run_single(ops: List[Op], x: Zonotope, pool: Optional[SymbolPool], counter: list) -> Zonotope:
    h = x
    for op in ops:
        if isinstance(op, AffineOp):
            h = zt.affine(h, op)
        elif isinstance(op, ReshapeOp):
            h = h.reshape(op.out_shape)
        elif isinstance(op, ActOp):
            h = _act_value(op, h, h.bounds(), pool, counter)
        else:
            raise TypeError(f"unexpected operation {op}")
    return h


def _diff_step(op_a: Op, op_b: Op, h: Zonotope, d: Zonotope, pool, counter):
    if isinstance(op_a, AffineOp):
        z = zt.affine(h, op_a)
        delta = zt.add(zt.affine(d, op_b, bias=False), zt.affine(h, op_a.difference(op_b)))
        return z, delta
    if isinstance(op_a, ReshapeOp):
        return h.reshape(op_a.out_shape), d.reshape(op_a.out_shape)
    if isinstance(op_a, ActOp):
        zp = zt.add(h, d)
        iz, idl, izp = h.bounds(), d.bounds(), zp.bounds()
        return (_act_value(op_a, h, iz, pool, counter),
                _act_diff(op_a, h, d, zp, iz, idl, izp, pool, counter))
    raise TypeError(f"unexpected operation {op_a}")


def _act_value(op: ActOp, z: Zonotope, iz: Interval, pool, counter) -> Zonotope:
    """Form for sigma(z) given the form of z and its interval iz."""
    everywhere = torch.ones_like(iz.lo, dtype=torch.bool)
    if pool is None:
        v = value_range(op.kind, op.slope, iz)
        return zt.fresh(v.center(), v.radius(), everywhere, None)
    if op.kind in ("relu", "leaky_relu"):
        a = op.slope if op.kind == "leaky_relu" else 0.0
        on, off = iz.lo >= 0, iz.hi <= 0
        unstable = ~(on | off)
        counter[0] += int(unstable.sum())
        lam, mid, rad = relaxation(op.kind, a, iz)
        zero = torch.zeros_like(lam)
        lam, mid, rad = (torch.where(unstable, t, zero) for t in (lam, mid, rad))
        relaxed = zt.add(zt.shift(zt.scale(z, lam), mid), zt.fresh(zero, rad, unstable, pool))
        off_form = zt.scale(z, a) if a else Zonotope.zeros(z.shape)
        return zt.select(on, z, zt.select(off, off_form, relaxed))
    counter[0] += int(everywhere.sum())
    lam, mid, rad = relaxation(op.kind, op.slope, iz)
    return zt.add(zt.shift(zt.scale(z, lam), mid), zt.fresh(torch.zeros_like(mid), rad, everywhere, pool))


def _act_diff(op: ActOp, z: Zonotope, delta: Zonotope, zp: Zonotope,
              iz: Interval, idl: Interval, izp: Interval, pool, counter) -> Zonotope:
    """Form for sigma(z + delta) - sigma(z)."""
    box = diff_range(op.kind, op.slope, iz, idl, izp)
    if pool is None:
        return zt.fresh(box.center(), box.radius(), torch.ones_like(box.lo, dtype=torch.bool), None)
    nothing = Zonotope.zeros(z.shape)
    zero = (idl.lo == 0) & (idl.hi == 0)
    if op.kind in ("relu", "leaky_relu"):
        a = op.slope if op.kind == "leaky_relu" else 0.0
        z_on, z_off = iz.lo >= 0, iz.hi <= 0
        p_on, p_off = izp.lo >= 0, izp.hi <= 0
        both_on, both_off = z_on & p_on, z_off & p_off
        on_off, off_on = z_on & p_off, z_off & p_on
        exact = zero | both_on | both_off | on_off | off_on
        counter[0] += int((~exact).sum())
        # exact cases: sigma(z') - sigma(z) with sigma linear on each side
        f_both_off = zt.scale(delta, a) if a else nothing          # a z' - a z
        f_on_off = zt.sub(zt.scale(zp, a), z)                        # a z' - z
        f_off_on = zt.sub(zp, zt.scale(z, a))                        # z' - a z
        f_box = zt.fresh(torch.where(exact, 0.0, box.center()), torch.where(exact, 0.0, box.radius()), ~exact, pool)
        out = zt.select(both_on, delta,
              zt.select(both_off, f_both_off,
              zt.select(on_off, f_on_off,
              zt.select(off_on, f_off_on, f_box))))
        return zt.select(zero, nothing, out)
    # tanh / sigmoid: sigma(z + delta) - sigma(z) = sigma'(xi) delta = kappa delta + (sigma'(xi) - kappa) delta
    slope = deriv_range(op.kind, hull(iz, izp))
    kappa = 0.5 * (slope.lo + slope.hi)
    rho = up(torch.maximum(slope.hi - kappa, kappa - slope.lo))
    band = up(rho * idl.mag())
    lin = zt.scale(delta, kappa)
    use_lin = 2 * (lin.radius() + band) <= box.hi - box.lo        # keep whichever enclosure is narrower
    counter[0] += int((~zero).sum())
    center = torch.where(use_lin, 0.0, box.center())
    radius = torch.where(use_lin, band, box.radius())
    out = zt.add(zt.select(use_lin, lin, nothing), zt.fresh(center, radius, ~zero, pool))
    return zt.select(zero, nothing, out)
