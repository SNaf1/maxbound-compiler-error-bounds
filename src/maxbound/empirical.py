"""Empirical checks: search for the largest difference by sampling and by a
gradient attack.

These only ever find a LOWER bound on the true maximum: an input where the two
models really differ by that much. They can refute a claimed bound (if they
beat it, the bound is wrong), but never prove one. The formal bound from
MaxBound must always be at least as large as anything found here.
"""
from __future__ import annotations

import copy
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .domain import as_box


def model_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Call the model and return its output tensor (accepts the common
    Hugging Face output objects that carry ``logits``)."""
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and len(out) == 1:
        return next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"cannot read a tensor from model output of type {type(out).__name__}")


def without_final_softmax(model: nn.Module) -> nn.Module:
    """A traced copy of the model with its final softmax removed, so the logits
    can be checked empirically (MaxBound(..., output="logits") bounds them)."""
    gm = torch.fx.symbolic_trace(copy.deepcopy(model).eval())
    out = next(n for n in gm.graph.nodes if n.op == "output")
    last = out.args[0]
    is_softmax = isinstance(last, torch.fx.Node) and (
        (last.op == "call_function" and last.target in (F.softmax, torch.softmax))
        or (last.op == "call_method" and last.target == "softmax")
        or (last.op == "call_module" and isinstance(gm.get_submodule(last.target), nn.Softmax)))
    if not is_softmax:
        raise ValueError("the model does not end with softmax")
    out.args = (last.args[0],)
    gm.graph.erase_node(last)
    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm.eval()


def _input_dtype(model: nn.Module) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


@torch.no_grad()
def differences(model: nn.Module, cl_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """||model(x) - cl_model(x)||_inf for each example in the batch x. Both
    models run in their own format; the subtraction is done in float64."""
    ya = model_output(model, x.to(_input_dtype(model))).double()
    yb = model_output(cl_model, x.to(_input_dtype(cl_model))).double()
    return (yb - ya).abs().flatten(1).max(1).values


def sampled_max_difference(model: nn.Module, cl_model: nn.Module, X: Any, n: int = 4096,
                           batch: int = 1024, seed: int = 0) -> Tuple[float, torch.Tensor]:
    """Largest difference over ``n`` random points of X (a quarter are corners)."""
    pts = as_box(X).sample(n, seed=seed)
    best, arg = -1.0, pts[0]
    for i in range(0, n, batch):
        diffs = differences(model, cl_model, pts[i:i + batch])
        j = int(diffs.argmax())
        if float(diffs[j]) > best:
            best, arg = float(diffs[j]), pts[i + j]
    return best, arg


def attack_max_difference(model: nn.Module, cl_model: nn.Module, X: Any, steps: int = 100,
                          restarts: int = 2, seed: int = 0) -> Tuple[float, torch.Tensor]:
    """Projected gradient ascent on the output difference.

    For every output coordinate i and sign s it runs ``restarts`` ascents that
    maximise s * (cl_model(x)_i - model(x)_i) inside X, with sign-gradient steps
    that shrink over time, and returns the largest L-inf difference seen."""
    box = as_box(X)
    lo, hi = box.float32_bounds()
    with torch.no_grad():
        n_out = model_output(model, box.center_float32().unsqueeze(0).to(_input_dtype(model))).numel()
    targets = torch.arange(n_out).repeat_interleave(2).repeat(restarts)
    signs = torch.tensor([1.0, -1.0]).repeat(n_out * restarts)
    g = torch.Generator().manual_seed(seed)
    t = torch.rand((targets.numel(), *box.shape), generator=g, dtype=torch.float64)
    x = (lo.double() + t * (hi.double() - lo.double())).float()
    span = (hi - lo).float()
    best, arg = -1.0, x[0]
    for step in range(steps + 1):
        with torch.no_grad():
            diffs = differences(model, cl_model, x)
            j = int(diffs.argmax())
            if float(diffs[j]) > best:
                best, arg = float(diffs[j]), x[j].clone()
        if step == steps:
            break
        x.requires_grad_(True)
        with torch.enable_grad():
            ya = model_output(model, x.to(_input_dtype(model))).flatten(1)
            yb = model_output(cl_model, x.to(_input_dtype(cl_model))).flatten(1)
            gap = (yb - ya).gather(1, targets[:, None]).squeeze(1) * signs
            grad, = torch.autograd.grad(gap.sum(), x)
        size = 0.25 * (1.0 - step / steps) + 0.002
        with torch.no_grad():
            x = torch.minimum(torch.maximum(x.detach() + size * span * grad.sign(), lo), hi)
    return best, arg
