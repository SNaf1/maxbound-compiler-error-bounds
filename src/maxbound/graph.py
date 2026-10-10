"""Turn a PyTorch model into a list of operations we can analyse.

We trace the model with torch.fx, which records the operations forward()
performs, and keep the ones that matter. maxbound supports models that form a
single chain (each operation consumes the previous result):

    nn.Linear / F.linear, nn.Conv2d / F.conv2d, eval-mode nn.BatchNorm1d/2d,
    ReLU, LeakyReLU (slope in [0, 1]), Tanh, Sigmoid,
    Flatten / view / reshape, Dropout and Identity (no-ops in eval mode),
    and a final Softmax over the last dimension.

Anything else (residual connections, LayerNorm, attention, pooling) raises
UnsupportedModelError naming the operation. To support a new layer type, add
a case to ``_convert`` and, for a new activation, its math to activations.py.

Weights are copied to float64. Converting float32 to float64 is exact, so the
analysis sees exactly the numbers the model stores.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from .errors import UnsupportedModelError
from .rounding import U64, check_range, gamma, inflate, up

Shape = Tuple[int, ...]


@dataclass
class AffineOp:
    """y = W x + b on one example x of shape ``in_shape``.

    kind "linear": W is a matrix (out, in).
    kind "conv2d": W is a convolution kernel, ``conv`` holds stride, padding...
    kind "scale":  W has one number per channel (eval-mode BatchNorm).

    ``weight_err`` / ``bias_err`` are radii of uncertainty around W and b. They
    are None for weights read from a model, and set only when we compute
    weights ourselves in float64 (BatchNorm folding, weight differences).
    """

    kind: str
    weight: torch.Tensor
    bias: torch.Tensor
    in_shape: Shape
    out_shape: Shape
    conv: Optional[dict] = None
    weight_err: Optional[torch.Tensor] = None
    bias_err: Optional[torch.Tensor] = None
    has_bias: bool = True
    bn_abs_mean: Optional[torch.Tensor] = None
    bn_abs_beta: Optional[torch.Tensor] = None
    bn_var: Optional[torch.Tensor] = None
    bn_eps: float = 0.0
    name: str = ""

    def fan_in(self) -> int:
        """Number of products summed for one output value."""
        if self.kind == "linear":
            return int(self.weight.shape[1])
        if self.kind == "conv2d":
            return int(self.weight[0].numel())
        return 1

    def _apply(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply the linear part with weights ``w`` to a batch x of shape (B, *in_shape)."""
        if self.kind == "linear":
            return x @ w.T
        if self.kind == "conv2d":
            if x.shape[0] == 0:
                return x.new_zeros((0, *self.out_shape))
            return F.conv2d(x, w, None, **self.conv)
        return x * w.view(-1, *([1] * (len(self.in_shape) - 1)))

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        """W x for a batch x of shape (B, *in_shape)."""
        return self._apply(x, self.weight)

    def abs_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Same map with every weight replaced by its absolute value, |W| x."""
        return self._apply(x, self.weight.abs())

    def err_linear(self, x: torch.Tensor) -> torch.Tensor:
        """|dW| x for a batch x, where dW is the radius around each weight."""
        return self._apply(x, self.weight_err)

    def per_output(self, v: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-channel vector (like the bias) to ``out_shape``."""
        if len(self.out_shape) == 1:
            return v
        return v.view(-1, *([1] * (len(self.out_shape) - 1))).expand(self.out_shape)

    def bias_full(self) -> torch.Tensor:
        """The bias broadcast to out_shape."""
        return self.per_output(self.bias)

    def difference(self, other: "AffineOp") -> "AffineOp":
        """The op with weights W_other - W_self and bias b_other - b_self.

        The subtraction is done in float64; its rounding error (at most
        2u times the result, zero when the result is zero) becomes part of
        weight_err / bias_err, together with any radii the two ops carry."""
        dw, db = other.weight - self.weight, other.bias - self.bias
        werr, berr = up(2 * U64 * dw.abs()), up(2 * U64 * db.abs())
        for op in (self, other):
            if op.weight_err is not None:
                werr = werr + op.weight_err
            if op.bias_err is not None:
                berr = berr + op.bias_err
        return AffineOp(self.kind, dw, db, self.in_shape, self.out_shape, self.conv,
                        inflate(werr, 4), inflate(berr, 4), True, name=f"diff({self.name})")


@dataclass
class ActOp:
    """Elementwise activation: "relu", "leaky_relu", "tanh" or "sigmoid"."""

    kind: str
    shape: Shape
    slope: float = 0.0
    name: str = ""


@dataclass
class ReshapeOp:
    in_shape: Shape
    out_shape: Shape
    name: str = ""


@dataclass
class SoftmaxOp:
    shape: Shape
    name: str = ""


Op = Union[AffineOp, ActOp, ReshapeOp, SoftmaxOp]


@dataclass
class Chain:
    """The operations of one model, in order, plus the float format it runs in."""

    ops: List[Op]
    dtype: torch.dtype
    in_shape: Shape
    out_shape: Shape


# ---------------------------------------------------------------------------
# building ops from modules and functions

def _linear_op(w: torch.Tensor, b: Optional[torch.Tensor], in_shape: Shape, out_shape: Shape, name: str) -> AffineOp:
    if len(in_shape) != 1:
        raise UnsupportedModelError(f"{name}: Linear on inputs with more than one feature dimension")
    W = w.detach().to(torch.float64).clone()
    B = b.detach().to(torch.float64).clone() if b is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    check_range(W, B, what=name)
    return AffineOp("linear", W, B, in_shape, out_shape, has_bias=b is not None, name=name)


def _conv_op(w, b, stride, padding, dilation, groups, in_shape, out_shape, name, padding_mode="zeros") -> AffineOp:
    if padding_mode != "zeros":
        raise UnsupportedModelError(f"{name}: only zero padding is supported")
    W = w.detach().to(torch.float64).clone()
    B = b.detach().to(torch.float64).clone() if b is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    check_range(W, B, what=name)
    conv = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    return AffineOp("conv2d", W, B, in_shape, out_shape, conv=conv, has_bias=b is not None, name=name)


def _bn_op(bn: nn.Module, shape: Shape, name: str) -> AffineOp:
    """Eval-mode BatchNorm as y = s * x + t per channel, with
    s = weight / sqrt(running_var + eps) and t = bias - running_mean * s.
    s and t are computed in float64; their rounding becomes weight_err / bias_err."""
    if bn.training or bn.running_mean is None:
        raise UnsupportedModelError(f"{name}: BatchNorm must be in eval mode with running statistics")
    mean, var = bn.running_mean.detach().double(), bn.running_var.detach().double()
    if bool((var < 0).any()) or bn.eps < 0:
        raise UnsupportedModelError(f"{name}: BatchNorm running_var and eps must be nonnegative")
    w = bn.weight.detach().double() if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias.detach().double() if bn.bias is not None else torch.zeros_like(mean)
    s = w / torch.sqrt(var + bn.eps)              # 3 roundings: add, sqrt, divide
    t = beta - mean * s                           # 2 roundings, plus the error carried by s
    s_err = up(s.abs() * gamma(4, U64))
    t_err = inflate(gamma(3, U64) * (beta.abs() + (mean * s).abs()) + mean.abs() * s_err, 4)
    check_range(s, t, what=name)
    return AffineOp("scale", s, t, shape, shape, weight_err=s_err, bias_err=t_err,
                    bn_abs_mean=mean.abs(), bn_abs_beta=beta.abs(), bn_var=var, bn_eps=float(bn.eps), name=name)


def _leaky(slope: float, shape: Shape, name: str) -> ActOp:
    if not 0.0 <= float(slope) <= 1.0:
        raise UnsupportedModelError(f"{name}: LeakyReLU slope must be in [0, 1]")
    return ActOp("leaky_relu", shape, float(slope), name)


def _softmax(dim: Any, shape: Shape, name: str) -> SoftmaxOp:
    if len(shape) != 1 or dim not in (-1, 1):
        raise UnsupportedModelError(f"{name}: softmax is supported only over the last dimension of a vector output")
    return SoftmaxOp(shape, name)


_MISSING = object()


def _arg(node: torch.fx.Node, index: int, name: str, default: Any = _MISSING) -> Any:
    if len(node.args) > index:
        return node.args[index]
    if name in node.kwargs:
        return node.kwargs[name]
    if default is _MISSING:
        raise UnsupportedModelError(f"{node.name}: missing argument {name}")
    return default


def _param(gm: torch.fx.GraphModule, node: Any) -> Optional[torch.Tensor]:
    if node is None:
        return None
    if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
        raise UnsupportedModelError("weights must be stored tensors of the model, not computed values")
    obj: Any = gm
    for part in node.target.split("."):
        obj = getattr(obj, part)
    return obj


def _convert(gm: torch.fx.GraphModule, node: torch.fx.Node, in_shape: Shape, out_shape: Shape) -> Optional[Op]:
    """Map one traced operation to an Op (None for operations that do nothing)."""
    name = node.name
    if node.op == "call_module":
        mod = gm.get_submodule(node.target)
        name = str(node.target)
        if isinstance(mod, nn.Linear):
            return _linear_op(mod.weight, mod.bias, in_shape, out_shape, name)
        if isinstance(mod, nn.Conv2d):
            return _conv_op(mod.weight, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups,
                            in_shape, out_shape, name, mod.padding_mode)
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return _bn_op(mod, in_shape, name)
        if isinstance(mod, nn.ReLU):
            return ActOp("relu", in_shape, name=name)
        if isinstance(mod, nn.LeakyReLU):
            return _leaky(mod.negative_slope, in_shape, name)
        if isinstance(mod, nn.Tanh):
            return ActOp("tanh", in_shape, name=name)
        if isinstance(mod, nn.Sigmoid):
            return ActOp("sigmoid", in_shape, name=name)
        if isinstance(mod, nn.Flatten):
            return ReshapeOp(in_shape, out_shape, name)
        if isinstance(mod, nn.Softmax):
            return _softmax(mod.dim, in_shape, name)
        if isinstance(mod, (nn.Identity, nn.Dropout)):
            return None
        raise UnsupportedModelError(f"module '{name}' of type {type(mod).__name__} is not supported")
    if node.op == "call_function":
        fn = node.target
        if fn is F.linear:
            w = _param(gm, _arg(node, 1, "weight"))
            b = _param(gm, _arg(node, 2, "bias", None))
            return _linear_op(w, b, in_shape, out_shape, name)
        if fn is F.conv2d:
            w = _param(gm, _arg(node, 1, "weight"))
            b = _param(gm, _arg(node, 2, "bias", None))
            return _conv_op(w, b, _arg(node, 3, "stride", 1), _arg(node, 4, "padding", 0),
                            _arg(node, 5, "dilation", 1), _arg(node, 6, "groups", 1), in_shape, out_shape, name)
        if fn in (F.relu, torch.relu):
            return ActOp("relu", in_shape, name=name)
        if fn is F.leaky_relu:
            return _leaky(_arg(node, 1, "negative_slope", 0.01), in_shape, name)
        if fn in (torch.tanh, F.tanh):
            return ActOp("tanh", in_shape, name=name)
        if fn in (torch.sigmoid, F.sigmoid):
            return ActOp("sigmoid", in_shape, name=name)
        if fn is torch.flatten:
            return ReshapeOp(in_shape, out_shape, name)
        if fn in (F.softmax, torch.softmax):
            if node.kwargs.get("dtype") is not None or len(node.args) > 3:
                raise UnsupportedModelError(f"{name}: softmax with a dtype argument is not supported")
            return _softmax(_arg(node, 1, "dim"), in_shape, name)
        if fn is F.dropout:
            if _arg(node, 2, "training", True):
                raise UnsupportedModelError(f"{name}: dropout is active; call model.eval()")
            return None
        raise UnsupportedModelError(f"function {getattr(fn, '__name__', fn)} ({name}) is not supported")
    if node.op == "call_method":
        method = node.target
        if method in ("relu", "tanh", "sigmoid"):
            return ActOp(method, in_shape, name=name)
        if method in ("view", "reshape", "flatten"):
            return ReshapeOp(in_shape, out_shape, name)
        if method == "softmax":
            if node.kwargs.get("dtype") is not None or len(node.args) > 2:
                raise UnsupportedModelError(f"{name}: softmax with a dtype argument is not supported")
            return _softmax(_arg(node, 1, "dim"), in_shape, name)
        if method == "contiguous":
            return None
        raise UnsupportedModelError(f"tensor method .{method}() ({name}) is not supported")
    raise UnsupportedModelError(f"unexpected graph node {node.op} {node.target}")


def _is_shape_query(node: torch.fx.Node, shape_nodes: set) -> bool:
    """Nodes like x.size(0) or x.shape[0] only read shapes; they are not data."""
    if node.op == "call_method" and node.target in ("size", "dim"):
        return True
    if node.op == "call_function" and node.target is getattr and node.args[1] in ("shape", "ndim"):
        return True
    return node.op == "call_function" and node.target is operator.getitem and node.args[0] in shape_nodes


def _per_example_shape(node: torch.fx.Node, dtype: Optional[torch.dtype] = None) -> Shape:
    """Shape of one example at this node; also checks the node keeps ``dtype``
    (an operation that casts, e.g. to float16, is outside the rounding model)."""
    meta = node.meta.get("tensor_meta")
    if meta is None or not hasattr(meta, "shape"):
        raise UnsupportedModelError(f"'{node.name}' does not produce a single tensor")
    if dtype is not None and meta.dtype != dtype:
        raise UnsupportedModelError(f"'{node.name}' produces {meta.dtype}, but the model runs in {dtype}")
    shape = tuple(meta.shape)
    if not shape or shape[0] != 2:
        raise UnsupportedModelError(f"'{node.name}' changes the batch dimension")
    return shape[1:]


def _single_output(arg: Any) -> Any:
    if isinstance(arg, (tuple, list)) and len(arg) == 1:
        return arg[0]
    if isinstance(arg, dict) and len(arg) == 1:
        return next(iter(arg.values()))
    return arg


def _model_dtype(model: nn.Module) -> torch.dtype:
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    dtypes |= {b.dtype for b in model.buffers() if b.is_floating_point()}
    if not dtypes:
        return torch.float32
    if len(dtypes) > 1 or next(iter(dtypes)) not in (torch.float32, torch.float64):
        raise UnsupportedModelError(f"the model must use float32 or float64 throughout, found {sorted(map(str, dtypes))}")
    return next(iter(dtypes))


def unwrap_compiled(model: nn.Module) -> Tuple[nn.Module, bool]:
    """A model returned by torch.compile wraps the original module in
    ``_orig_mod`` and runs Inductor-generated kernels for it. We analyse the
    original module's graph, under the assumption that Inductor keeps the
    real-number function and changes only rounding: it fuses operations,
    reorders sums and picks kernels. On Windows it compiles without fast-math
    flags; elsewhere it adds -fno-unsafe-math-optimizations unless
    TORCHINDUCTOR_CPP_ENABLE_UNSAFE_MATH_OPT_FLAG=1 (torch/_inductor/cpp_builder.py,
    torch 2.14). The rounding is what fp_execution bounds, for any summation order.
    Returns (module to analyse, whether it was compiled)."""
    inner = getattr(model, "_orig_mod", None)
    if isinstance(inner, nn.Module):
        return inner, True
    return model, False


def extract(model: nn.Module, example: torch.Tensor) -> Chain:
    """Trace ``model`` and return its chain of operations.

    ``example`` is one input without a batch dimension; we run the traced graph
    on a batch of two copies to learn every intermediate shape (two, so that a
    reshape that would mix examples is caught)."""
    model, _ = unwrap_compiled(model)
    if model.training:
        raise ValueError("put the model in eval mode first (model.eval())")
    dtype = _model_dtype(model)
    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception as exc:  # torch.fx raises many different exception types
        raise UnsupportedModelError(f"torch.fx could not trace the model: {exc}") from exc
    batch = example.detach().to(dtype).unsqueeze(0).repeat(2, *([1] * example.dim()))
    with torch.no_grad():
        ShapeProp(gm).propagate(batch)

    ops: List[Op] = []
    shape_nodes: set = set()
    current: Optional[torch.fx.Node] = None
    first: Optional[torch.fx.Node] = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if current is None:
                current = first = node
            elif node.users:
                raise UnsupportedModelError("the model uses more than one input tensor")
            continue
        if node.op == "get_attr":
            continue
        if node.op == "output":
            if _single_output(node.args[0]) is not current:
                raise UnsupportedModelError("the model output is not the end of a single chain of operations")
            break
        if _is_shape_query(node, shape_nodes):
            shape_nodes.add(node)
            continue
        data_inputs = [n for n in node.all_input_nodes if n.op != "get_attr" and n not in shape_nodes]
        if data_inputs != [current]:
            raise UnsupportedModelError(
                f"operation '{node.name}' does not consume exactly the previous result "
                "(branches, skip connections and multi-input operations are not supported)"
            )
        op = _convert(gm, node, _per_example_shape(current), _per_example_shape(node, dtype))
        if op is not None:
            ops.append(op)
        current = node
    if current is None or first is None:
        raise UnsupportedModelError("the model has no input")
    for i, op in enumerate(ops):
        if isinstance(op, SoftmaxOp) and i != len(ops) - 1:
            raise UnsupportedModelError("softmax is supported only as the last operation")
    return Chain(ops, dtype, _per_example_shape(first), _per_example_shape(current))


# ---------------------------------------------------------------------------
# preparing two chains for the differential analysis

def fold_batchnorm(ops: List[Op]) -> List[Op]:
    """Merge every BatchNorm ("scale" op) into the Linear/Conv right before it.

    In real arithmetic s * (W x + b) + t = (s W) x + (s b + t), so this does not
    change the function. It makes "Linear + BatchNorm" in the original line up
    with the single fused Linear a compiler produces. The float64 rounding of
    s W and s b + t is recorded in weight_err / bias_err."""
    out: List[Op] = []
    for op in ops:
        prev = out[-1] if out else None
        if (isinstance(op, AffineOp) and op.kind == "scale"
                and isinstance(prev, AffineOp) and prev.kind in ("linear", "conv2d")):
            out[-1] = _fold(prev, op)
        else:
            out.append(op)
    return out


def _fold(prev: AffineOp, sc: AffineOp) -> AffineOp:
    shape = (-1,) + (1,) * (prev.weight.dim() - 1)
    s, s_abs = sc.weight.view(shape), sc.weight.abs()
    W = prev.weight * s
    b = prev.bias * sc.weight + sc.bias
    werr = 2 * U64 * W.abs() + prev.weight.abs() * sc.weight_err.view(shape)
    if prev.weight_err is not None:
        werr = werr + prev.weight_err * s.abs()
    berr = 2 * U64 * ((prev.bias * sc.weight).abs() + b.abs()) + prev.bias.abs() * sc.weight_err + sc.bias_err
    if prev.bias_err is not None:
        berr = berr + prev.bias_err * s_abs
    return AffineOp(prev.kind, W, b, prev.in_shape, prev.out_shape, prev.conv,
                    inflate(werr, 4), inflate(berr, 6), True, name=f"{prev.name}+{sc.name}")


def alignment_problem(a: List[Op], b: List[Op]) -> Optional[str]:
    """None if the two chains have the same structure (same op types, shapes and
    activation settings, weights may differ); otherwise a reason."""
    if len(a) != len(b):
        return f"different number of operations ({len(a)} vs {len(b)})"
    for i, (x, y) in enumerate(zip(a, b)):
        if type(x) is not type(y):
            return f"operation {i}: {type(x).__name__} vs {type(y).__name__}"
        if isinstance(x, AffineOp):
            if (x.kind, x.in_shape, x.out_shape, x.conv) != (y.kind, y.in_shape, y.out_shape, y.conv):
                return f"operation {i}: affine layers differ in kind, shape or convolution settings"
            if x.weight.shape != y.weight.shape:
                return f"operation {i}: weight shapes differ"
        elif isinstance(x, ActOp) and (x.kind, x.slope, x.shape) != (y.kind, y.slope, y.shape):
            return f"operation {i}: activations differ ({x.kind} vs {y.kind})"
        elif isinstance(x, ReshapeOp) and x.out_shape != y.out_shape:
            return f"operation {i}: reshapes differ"
    return None
