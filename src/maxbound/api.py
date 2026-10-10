"""The public entry point: ``bound = MaxBound(model, cl_model, X)``.

Steps:
 1. turn X into a Box (domain.py);
 2. trace both models into chains of operations (graph.py);
 3. bound the output difference in exact real arithmetic with every applicable
    analysis (analysis.py) and keep the smallest result per output;
 4. add, for each model, a bound on how far its float32 execution can drift
    from its real-number function (fp_execution.py);
 5. return a Bound: a float that also carries the details.

The norm is L-infinity over output coordinates: the bound covers the largest
absolute difference in any single output value, for every input in X.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .analysis import METHODS, Result, run
from .config import MaxBoundConfig
from .domain import as_box
from .errors import UnsupportedModelError
from .fp_execution import Allowance, execution_allowance
from .graph import SoftmaxOp, alignment_problem, extract, fold_batchnorm, unwrap_compiled
from .intervals import Interval
from .rounding import down, inflate, up
from .softmax import difference_bound, pair_bounds, prob_range

#: Messages go to the "maxbound" logger. Warnings show by default; to see the
#: step-by-step messages, call logging.basicConfig(level=logging.INFO).
logger = logging.getLogger(__name__)

ASSUMPTIONS = [
    "Norm: L-infinity over output coordinates; the bound covers every input in X (a box after outward rounding).",
    "Each model computes exactly the chain of operations traced by torch.fx (single chain, eval mode).",
    "Both models run on the CPU in IEEE float32 (or float64) with round-to-nearest; matrix products and "
    "convolutions use full precision (no TF32/bfloat16), in any summation order, with or without fused "
    "multiply-add; every operation keeps the model's dtype.",
    "float32 tanh, sigmoid and exp are accurate to 16 units of roundoff; float64 tanh, sigmoid and exp to 2^-45 "
    "relative (both checked by tests on grids).",
    "Eval-mode BatchNorm executes as x * alpha + (beta - mean * alpha) with alpha = weight / sqrt(var + eps), "
    "each step rounded once and eps used exactly or rounded to the model's format; softmax executes as "
    "exp(x - max) / sum.",
    "No overflow (checked during the analysis).",
    "A model produced by torch.compile computes the same real-number function as its original module "
    "(Inductor fuses and reorders operations without unsafe fast-math); only its rounding differs.",
]


class Bound(float):
    """The bound as a plain number, plus how it was obtained.

    ``float(bound)`` is a sound upper bound on max over x in X of
    ||model(x) - cl_model(x)||_inf under the listed ``assumptions``."""

    def __new__(cls, value: float, **info: Any) -> "Bound":
        obj = super().__new__(cls, value)
        obj.__dict__.update(info)
        return obj

    def __repr__(self) -> str:
        return (f"Bound({float(self):.6g}, real={self.real_bound:.3g}, "
                f"fp_allowance={self.fp_allowance:.3g}, best={self.best_method})")

    def summary(self) -> str:
        """A readable multi-line report of the bound and how it was obtained."""
        lines = [f"bound (L-inf, {self.output_kind}): {float(self):.6g}",
                 f"  real-arithmetic part: {self.real_bound:.6g}  (best analysis: {self.best_method})",
                 f"  floating-point allowance (both models): {self.fp_allowance:.6g}"]
        for name, value in self.method_bounds.items():
            lines.append(f"  {name:>18}: {value:.6g}")
        if self.alignment_problem:
            lines.append(f"  differential analysis skipped: {self.alignment_problem}")
        if self.same_prediction is not None:
            lines.append(f"  certified: both models' {self.output_kind} rank class {self.same_prediction} "
                         "strictly first for every input in X")
        return "\n".join(lines)


def check_runtime(*models: nn.Module) -> None:
    """Refuse settings that would break assumption A1. Checked: the float32
    matmul precision; every fp32_precision switch PyTorch exposes (global and
    oneDNN matmul / conv / rnn); CPU autocast; the device of every parameter and
    buffer; and, for torch.compile models, Inductor's unsafe-math switches (both
    the live config and the environment variables)."""
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("torch.set_float32_matmul_precision must be 'highest' for a sound bound")
    mkldnn = getattr(torch.backends, "mkldnn", None)
    for owner, label in ((torch.backends, "torch.backends"),
                         (getattr(mkldnn, "matmul", None), "torch.backends.mkldnn.matmul"),
                         (getattr(mkldnn, "conv", None), "torch.backends.mkldnn.conv"),
                         (getattr(mkldnn, "rnn", None), "torch.backends.mkldnn.rnn")):
        if getattr(owner, "fp32_precision", "none") not in ("none", "ieee"):
            raise RuntimeError(f"{label}.fp32_precision must be 'ieee' (or 'none') for a sound bound")
    try:
        autocast = torch.is_autocast_enabled("cpu")
    except TypeError:                       # older signature without a device argument
        autocast = torch.is_autocast_cpu_enabled()
    if autocast:
        raise RuntimeError("CPU autocast is active; maxbound covers float32/float64 execution only")
    for m in models:
        for t in list(m.parameters()) + list(m.buffers()):
            if t.device.type != "cpu":
                raise RuntimeError("maxbound's rounding model covers CPU execution only")
    if any(unwrap_compiled(m)[1] for m in models):
        unsafe = (os.environ.get("TORCHINDUCTOR_CPP_ENABLE_UNSAFE_MATH_OPT_FLAG") == "1"
                  or os.environ.get("TORCHINDUCTOR_USE_FAST_MATH") == "1")
        try:
            from torch._inductor import config as inductor_config
            unsafe = (unsafe or bool(inductor_config.cpp.enable_unsafe_math_opt_flag)
                      or bool(getattr(inductor_config, "use_fast_math", False)))
        except (ImportError, AttributeError):
            pass
        if unsafe:
            raise RuntimeError("torch.compile is set to use unsafe fast-math; the rounding model does not cover it")


def _real_bound(res: Result, softmax: bool) -> torch.Tensor:
    if softmax:
        return difference_bound(res.diff, res.a, res.b)
    return res.diff.bounds().mag().flatten()


def _allowance(ops, box: Interval, dtype: torch.dtype, fp_model: str, n_out: int) -> Allowance:
    if fp_model == "real":
        zero = torch.zeros(n_out, dtype=torch.float64)
        return Allowance(zero, zero)
    fmt = dtype if fp_model == "auto" else getattr(torch, fp_model)
    return execution_allowance(ops, box, fmt)


def _certify(results: Dict[str, Result], alw_a: torch.Tensor, alw_b: torch.Tensor,
             softmax: bool) -> Optional[int]:
    """A class c such that, for every input in X, both models' executed bounded
    outputs (probabilities if ``softmax``, else the outputs before any softmax)
    put c strictly above every other class; None if we cannot show one.

    Without softmax we use pairwise lower bounds of v_c - v_j. With softmax we
    compare probability ranges directly: rounding can turn distinct logits into
    equal probabilities, so a logit ordering is not enough."""
    res = results.get("zonotope") or results.get("zonotope-separate")
    if res is None or res.a.c.dim() != 1 or res.a.c.numel() < 2:
        return None
    n = res.a.c.numel()
    eye = torch.eye(n, dtype=torch.bool)
    ok = torch.ones(n, dtype=torch.bool)
    for net, alw in ((res.a, alw_a), (res.b, alw_b)):
        if softmax:
            p = prob_range([pair_bounds(net)])
            low, high = down(p.lo - alw), up(p.hi + alw)        # executed probability range
            ok &= ((low[:, None] > high[None, :]) | eye).all(1)
        else:
            margin = pair_bounds(net).lo                         # [i, j]: lower bound of v_i - v_j
            need = inflate(alw[:, None] + alw[None, :], 2)
            ok &= ((margin > need) | eye).all(1)
    winners = ok.nonzero()
    return int(winners[0]) if winners.numel() else None


def MaxBound(model: nn.Module, cl_model: nn.Module, X: Any,
             config: Any = None, **overrides: Any) -> Bound:
    """Sound upper bound on max over x in X of ||model(x) - cl_model(x)||_inf.

    model, cl_model: PyTorch modules in eval mode (a Hugging Face model is a
        PyTorch module); cl_model is typically cl_func(model).
    X: a Box, a LinfBall, a dict config or a (lower, upper) pair, describing the
        inputs of one example (no batch dimension).
    config / overrides: see MaxBoundConfig (method, output, fp_model, ...).
    """
    cfg = MaxBoundConfig.build(config, **overrides)
    box = as_box(X)
    if cfg.check_runtime:
        check_runtime(model, cl_model)
    example = box.interval().center()      # only used to learn shapes; extract casts it to each model's dtype
    chain_a, chain_b = extract(model, example), extract(cl_model, example)
    if chain_a.in_shape != tuple(box.shape) or chain_b.in_shape != tuple(box.shape):
        raise ValueError(f"X has shape {tuple(box.shape)} but the models take {chain_a.in_shape}")
    ends_a = bool(chain_a.ops) and isinstance(chain_a.ops[-1], SoftmaxOp)
    ends_b = bool(chain_b.ops) and isinstance(chain_b.ops[-1], SoftmaxOp)
    if ends_a != ends_b:
        raise UnsupportedModelError("only one of the two models ends with softmax")
    use_softmax = ends_a and cfg.output == "model"
    body_a = chain_a.ops[:-1] if ends_a else chain_a.ops
    body_b = chain_b.ops[:-1] if ends_b else chain_b.ops
    ops_a, ops_b = fold_batchnorm(body_a), fold_batchnorm(body_b)
    problem = alignment_problem(ops_a, ops_b)
    logger.info("traced %d operations (%s) and %d operations (%s); X has shape %s",
                len(chain_a.ops), chain_a.dtype, len(chain_b.ops), chain_b.dtype, tuple(box.shape))
    if problem is not None:
        logger.warning("the two models do not line up (%s); using the separate analyses only, "
                       "which are sound but looser", problem)

    iv = box.interval()
    names = list(METHODS) if cfg.method == "best" else [cfg.method]
    per_method: Dict[str, torch.Tensor] = {}
    results: Dict[str, Result] = {}
    for name in names:
        if METHODS[name][1] and problem is not None:
            continue
        res = run(ops_a, ops_b, iv, name)
        per_method[name], results[name] = _real_bound(res, use_softmax), res
        detail = f" ({res.relaxed} neurons approximated)" if name.startswith("zonotope") else ""
        logger.info("%s analysis: real-arithmetic bound %.6g%s", name, float(per_method[name].max()), detail)
    if not per_method:
        raise UnsupportedModelError(f"the models do not line up for differential analysis: {problem}")
    real = torch.stack(list(per_method.values())).min(0).values
    best = min(per_method, key=lambda k: float(per_method[k].max()))

    n_out = real.numel()
    exec_a = body_a + ([chain_a.ops[-1]] if use_softmax else [])
    exec_b = body_b + ([chain_b.ops[-1]] if use_softmax else [])
    alw_a = _allowance(exec_a, iv, chain_a.dtype, cfg.fp_model, n_out)
    alw_b = _allowance(exec_b, iv, chain_b.dtype, cfg.fp_model, n_out)
    total = inflate(real + alw_a.output + alw_b.output, 3)
    same = _certify(results, alw_a.output, alw_b.output, use_softmax)
    logger.info("rounding allowance (fp_model=%s): original %.3g, compiled %.3g",
                cfg.fp_model, float(alw_a.output.max()), float(alw_b.output.max()))
    logger.info("MaxBound = %.6g (real part %.6g from the %s analysis); same-class certificate: %s",
                float(total.max()), float(real.max()), best, same)
    return Bound(
        float(total.max()),
        norm="linf",
        output_kind="probabilities" if use_softmax else "outputs",
        real_bound=float(real.max()),
        fp_allowance=float((alw_a.output + alw_b.output).max()),
        per_output=total.tolist(),
        method_bounds={k: float(v.max()) for k, v in per_method.items()},
        best_method=best,
        alignment_problem=problem,
        relaxed_neurons={k: r.relaxed for k, r in results.items() if k.startswith("zonotope")},
        same_prediction=same,
        dtypes=(str(chain_a.dtype), str(chain_b.dtype)),
        fp_model=cfg.fp_model,
        assumptions=list(ASSUMPTIONS),
    )
