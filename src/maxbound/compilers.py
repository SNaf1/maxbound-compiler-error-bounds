"""Example compile passes ``cl_func``: each takes a model and returns a changed copy.

They are the "compiler" side of ``cl_model = cl_func(model)`` in the examples
and tests. The original model is never modified.

* ``quantize_weights``: symmetric fake quantization of weight matrices and
  convolution kernels to ``bits`` bits (weights are stored back in float32,
  activations are not quantized).
* ``cast_weights``: round every parameter to float16 or bfloat16 and back.
* ``prune_magnitude``: set the smallest weights of each matrix to zero.
* ``fuse_batchnorm``: fold eval-mode BatchNorm into the Linear/Conv before it,
  the rewrite inference compilers apply, using PyTorch's own fusion helpers.
* ``torch_compile``: PyTorch's real compiler (TorchDynamo + Inductor), which
  generates and compiles C++ kernels for the CPU.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval


def quantize_weights(model: nn.Module, bits: int = 8, per_channel: bool = True) -> nn.Module:
    """Round every parameter with 2 or more dimensions to a symmetric grid:
    w_q = scale * clamp(round(w / scale), -qmax, qmax), qmax = 2**(bits-1) - 1,
    scale = max|w| / qmax per output channel (or per tensor). Biases and
    normalisation parameters are left as they are."""
    if bits < 2:
        raise ValueError("bits must be at least 2")
    q = copy.deepcopy(model).eval()
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in q.parameters():
            if p.dim() < 2:
                continue
            dims = tuple(range(1, p.dim()))
            amax = p.abs().amax(dim=dims, keepdim=True) if per_channel else p.abs().max()
            scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
            p.copy_(torch.clamp(torch.round(p / scale), -qmax, qmax) * scale)
    return q


def cast_weights(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """Round every floating parameter to ``dtype`` and back to its own format."""
    q = copy.deepcopy(model).eval()
    with torch.no_grad():
        for p in q.parameters():
            if p.is_floating_point():
                p.copy_(p.to(dtype).to(p.dtype))
    return q


def prune_magnitude(model: nn.Module, sparsity: float = 0.5) -> nn.Module:
    """Zero the fraction ``sparsity`` of smallest-magnitude entries of every
    parameter with 2 or more dimensions."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    q = copy.deepcopy(model).eval()
    with torch.no_grad():
        for p in q.parameters():
            if p.dim() < 2 or sparsity == 0.0:
                continue
            k = int(sparsity * p.numel())
            if k:
                cutoff = p.abs().flatten().kthvalue(k).values
                p.mul_((p.abs() > cutoff).to(p.dtype))
    return q


def torch_compile(model: nn.Module, **kwargs) -> nn.Module:
    """``torch.compile`` with the default Inductor backend, applied to a copy.

    Inductor fuses operations and generates C++ kernels, so the compiled model is
    not bit-for-bit identical to the original. It needs a C++ compiler; on
    Windows, run inside a Visual Studio developer environment (vcvars64.bat).
    See graph.unwrap_compiled for how MaxBound analyses the result."""
    return torch.compile(copy.deepcopy(model).eval(), **kwargs)


def fuse_batchnorm(model: nn.Module) -> nn.Module:
    """Return a traced copy where every Linear->BatchNorm1d and Conv2d->BatchNorm2d
    pair is replaced by one fused layer (torch.nn.utils.fusion)."""
    gm = torch.fx.symbolic_trace(copy.deepcopy(model).eval())
    modules = dict(gm.named_modules())
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not isinstance(modules[node.target], (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        prev = node.args[0]
        if not (isinstance(prev, torch.fx.Node) and prev.op == "call_module" and len(prev.users) == 1):
            continue
        layer, bn = modules[prev.target], modules[node.target]
        if isinstance(layer, nn.Linear) and isinstance(bn, nn.BatchNorm1d):
            fused = fuse_linear_bn_eval(layer, bn)
        elif isinstance(layer, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            fused = fuse_conv_bn_eval(layer, bn)
        else:
            continue
        parent_name, _, child = prev.target.rpartition(".")
        setattr(gm.get_submodule(parent_name) if parent_name else gm, child, fused)
        node.replace_all_uses_with(prev)
        gm.graph.erase_node(node)
    gm.graph.lint()
    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm.eval()
