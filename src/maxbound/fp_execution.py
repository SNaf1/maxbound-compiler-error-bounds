"""How far can the actual floating-point execution of a network drift from the
exact real-number function that the analysis reasons about?

The analysis (analysis.py) bounds |compiled(x) - original(x)| in exact real
arithmetic. The models themselves run in float32 (or float64), so each one is
off from its real-number function by a rounding error. This module bounds that
error for one network over the whole input region, layer by layer:

    a_k = bound on |executed value - real value| after layer k

* affine layer y = W x + b:  a_new = |W| a + gamma_{n+1} (|W| |x| + |b|)
  The first term carries the error that arrived; the second is the rounding of
  this layer's own dot products (Higham 2002, Section 3.1), for any summation
  order, with or without fused multiply-add. |x| is bounded by the real range
  of x (a zonotope analysis of this network alone) plus a.
* eval-mode BatchNorm: see ``_batchnorm_terms``.
* ReLU: exact in floating point and 1-Lipschitz, so a_new = a.
* other activations: a_new = L a + (own rounding), with L the Lipschitz constant.
* final softmax: see ``_softmax_terms``.

The final bound is real-arithmetic bound + allowance(original) + allowance(compiled)
(triangle inequality). Assumptions:
  A1 matrix products and convolutions run in the model's format (no TF32 or
     bfloat16 shortcuts) with round-to-nearest (enforced by api.check_runtime);
  A2 tanh, sigmoid and exp have relative error at most TRANS_U * u in float32 and
     LIBM_REL in float64 (tested on grids);
  A3 eval-mode BatchNorm executes as x * alpha + (beta - mean * alpha) with
     alpha = weight / sqrt(var + eps), each step rounded once, where eps is used
     either exactly or rounded to the model's format (see ``_batchnorm_terms``);
  A4 softmax is computed as exp(x - max(x)) divided by the sum of those exps;
  A5 no overflow (checked here).
Underflow, including flush-to-zero, adds at most TINY = 2**-126 (float32) to any
single rounded operation. Where such an error is later multiplied by another
value, the product is accounted for (BatchNorm's alpha times x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from . import zonotope as zt
from .activations import LIBM_REL
from .analysis import _act_value
from .errors import NumericalInconsistency
from .graph import ActOp, AffineOp, Op, ReshapeOp, SoftmaxOp
from .intervals import Interval
from .rounding import down, gamma, inflate, unit_roundoff, up
from .softmax import max_pq, pair_bounds, prob_range
from .zonotope import SymbolPool, Zonotope

#: Assumed relative error of float32 tanh / sigmoid / exp, in units of u.
TRANS_U = 16
_TINY = {torch.float32: 2.0 ** -126, torch.float64: 2.0 ** -1022}
_LARGEST = {torch.float32: 3.0e38, torch.float64: 1.0e307}
_LIPSCHITZ = {"relu": 1.0, "leaky_relu": 1.0, "tanh": 1.0, "sigmoid": 0.25}


def library_rel(dtype: torch.dtype) -> float:
    """Assumed relative error of tanh / sigmoid / exp in the given format (A2)."""
    return TRANS_U * unit_roundoff(torch.float32) if dtype == torch.float32 else LIBM_REL


@dataclass
class Allowance:
    output: torch.Tensor    # per output coordinate: bound on |executed - real|
    logits: torch.Tensor    # the same just before a final softmax (equal to output if none)


def execution_allowance(ops: List[Op], box: Interval, dtype: torch.dtype) -> Allowance:
    """Bound on |executed output - real output| over the box, for one network
    running in ``dtype`` (float32 or float64)."""
    u, tiny, lib = unit_roundoff(dtype), _TINY[dtype], library_rel(dtype)
    pool = SymbolPool()
    z = Zonotope.from_box(box, pool)                     # zonotope analysis of the real values
    a = torch.zeros(box.shape, dtype=torch.float64)
    logits = None
    for op in ops:
        iv = z.bounds()
        if isinstance(op, AffineOp):
            mag_in = inflate(iv.mag() + a, 2)
            n = op.fan_in() + 1
            size = op.abs_linear(mag_in.unsqueeze(0))[0] + op.bias_full().abs()
            if bool((size > _LARGEST[dtype] / 4).any()):
                raise NumericalInconsistency(f"{op.name}: values could overflow {dtype}")
            if op.kind == "scale":
                own, carry = _batchnorm_terms(op, mag_in, u, tiny, dtype)
                a = inflate(carry * a + own, 8)
            else:
                own = gamma(n, u) * size + n * tiny
                a = inflate(op.abs_linear(a.unsqueeze(0))[0] + own, n + 6)
            z = zt.affine(z, op)
        elif isinstance(op, ActOp):
            lip = _LIPSCHITZ[op.kind]
            in_mag = inflate(iv.mag() + a, 2)
            z = _act_value(op, z, iv, pool, [0])
            if op.kind == "relu":
                own = torch.zeros_like(a)
            elif op.kind == "leaky_relu":
                # PyTorch first rounds the Python slope to the tensor's format, then multiplies.
                stored = float(torch.tensor(op.slope, dtype=dtype))
                own = (u * stored + abs(stored - op.slope) * (1 + 4 * 2.0 ** -53)) * in_mag + tiny
            else:
                out_mag = inflate(z.bounds().mag() + lip * a, 2)
                own = lib * out_mag + tiny
            a = inflate(lip * a + own, 4)
        elif isinstance(op, ReshapeOp):
            z, a = z.reshape(op.out_shape), a.reshape(op.out_shape)
        elif isinstance(op, SoftmaxOp):
            logits = a
            a = _softmax_terms(pair_bounds(z), a.flatten(), u, tiny, lib)
        else:
            raise TypeError(f"unexpected operation {op}")
    return Allowance(a.flatten(), (logits if logits is not None else a).flatten())


def _batchnorm_terms(op: AffineOp, mag_in: torch.Tensor, u: float, tiny: float, dtype: torch.dtype):
    """Rounding of eval-mode BatchNorm executed as y = x * alpha + (b - mean * alpha)
    with alpha = w / sqrt(var + eps) (A3). Returns (own error, factor for the
    error a that arrives with x).

    1. The denominator. PyTorch may add eps exactly or after rounding it to the
       model's format (eps*), and the sum var + eps is rounded once more. So the
       executed denominator d lies in [S_lo (1 - u) - TINY, S_hi (1 + u) + TINY],
       with S over both choices of eps. If d could be <= 0 we refuse.
    2. alpha versus the real s = w / sqrt(var + eps): alpha / s = sqrt(D / d) times
       at most 4 roundings (sqrt, divide or reciprocal-and-multiply), so
       |alpha - s| <= e |s| + TINY with e = r + gamma_4 (1 + r) and
       r = max |sqrt(D / d) - 1| over the range of d.
    3. Then y - y_real = (alpha - s)(x - mean) plus the rounding of mean * alpha,
       the subtraction, x * alpha and the sum: at most
       e |s| (|x| + |mean|) + gamma_4 (|alpha| (|x| + |mean|) + |b|) + TINY (|x| + |mean| + 3),
       using |alpha| <= (1 + e) |s| + TINY. The incoming error is multiplied by |alpha|."""
    var, eps = op.bn_var, op.bn_eps
    eps_stored = float(torch.tensor(eps, dtype=dtype))
    s_lo = down(torch.minimum(var + eps, var + eps_stored))
    s_hi = up(torch.maximum(var + eps, var + eps_stored))
    d_lo, d_hi = down(s_lo * (1.0 - u)) - tiny, up(s_hi * (1.0 + u)) + tiny
    if bool((d_lo <= 0).any()):
        raise NumericalInconsistency(f"{op.name}: BatchNorm's var + eps could round to zero or below")
    real_lo, real_hi = down(var + eps), up(var + eps)
    r = torch.maximum(up(torch.sqrt(real_hi / d_lo)) - 1.0, 1.0 - down(torch.sqrt(real_lo / d_hi)))
    r = up(r.clamp(min=0.0) * (1.0 + 2.0 ** -40) + 2.0 ** -1000)
    g4 = gamma(4, u)
    e = r + g4 * (1.0 + r)
    s_abs = op.weight.abs()
    alpha = op.per_output(up((1.0 + e) * s_abs) + tiny)
    mean, beta = op.per_output(op.bn_abs_mean), op.per_output(op.bn_abs_beta)
    own = (op.per_output(e * s_abs) * (mag_in + mean)
           + g4 * (alpha * (mag_in + mean) + beta)
           + tiny * (mag_in + mean + 3.0))
    return own, alpha


def _softmax_terms(pairs: Interval, a: torch.Tensor, u: float, tiny: float, lib: float) -> torch.Tensor:
    """Allowance after a final softmax. ``pairs[i, j]`` encloses the real
    logit difference v_i - v_j; the executed logits are off by at most ``a``.

    Carried error: by the softmax mean value bound (softmax.py) the
    probabilities move by at most 2 max p(1 - p) * max(a), with p ranging over
    the probabilities of all logits within a of the real ones.
    Own rounding: computing exp(x - max), the sum and the division changes p_i by
    a relative amount of at most 1.03 * kappa, kappa = (2 D + 6) u + 2 lib + gamma_n,
    where D bounds max_j x_j - min_j x_j for the executed logits, lib is the
    relative error of exp (A2) and kappa <= 0.01 (derivation in docs/DESIGN.md, section 8)."""
    slack = a[:, None] + a[None, :]
    widened = Interval(down(pairs.lo - slack), up(pairs.hi + slack))
    p = prob_range([widened])
    carried = up(2.0 * max_pq(p) * a.max())
    spread = float(widened.hi.max())
    kappa = (2.0 * spread + 6) * u + 2.0 * lib + gamma(a.numel(), u)
    if kappa > 0.01:
        raise NumericalInconsistency("softmax rounding model needs (2D + 6) u + 2 lib + gamma_n <= 0.01")
    own = up(p.hi * kappa * 1.03) + tiny
    return inflate(carried + own, 3)
