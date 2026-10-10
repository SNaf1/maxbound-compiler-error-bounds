"""Cases small enough to check by hand. fp_model="real" gives the pure
real-arithmetic bound, so the numbers can be compared with pen-and-paper values."""
import copy

import pytest
import torch
import torch.nn as nn

from maxbound import Box, MaxBound

from conftest import make_mlp


def _pair(w_a, w_b, relu):
    """Two one-layer models without bias, with the given weight rows."""
    out = []
    for w in (w_a, w_b):
        lin = nn.Linear(len(w), 1, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor([w]))
        out.append((nn.Sequential(lin, nn.ReLU()) if relu else nn.Sequential(lin)).eval())
    return out


def test_linear_case():
    """x1 + x2 versus 1.1 x1 + 0.9 x2 on [0, 1]^2. The difference is
    0.1 x1 - 0.1 x2, largest in size at x = (1, 0) or (0, 1), where it is 0.1
    (exactly: float32(1.1) - 1 = 0.10000002384...). Bounding each model alone
    gives [0, 2] for both, hence a useless 2.0 for the naive method."""
    a, b = _pair([1.0, 1.0], [1.1, 0.9], relu=False)
    bound = MaxBound(a, b, Box(torch.zeros(2), torch.ones(2)), fp_model="real")
    exact = float(torch.tensor(1.1, dtype=torch.float32)) - 1.0
    m = bound.method_bounds
    for name in ("zonotope", "interval", "zonotope-separate"):
        assert m[name] == pytest.approx(exact, abs=1e-12)
    assert m["interval-separate"] == pytest.approx(2.0, abs=1e-12)
    assert float(bound) == pytest.approx(exact, abs=1e-12)


def test_relu_case():
    """relu(x) versus relu(1.5 x) on [-1, 1]. True maximum: 0.5 at x = 1.
    Differential: z in [-1, 1], delta = 0.5 x in [-0.5, 0.5]; the four corners of
    relu(z + delta) - relu(z) give [-0.5, 0.5], so 0.5 (exact).
    Separate zonotopes: DeepZ relaxes each ReLU with its own noise symbol,
    0.5x + 0.25 + 0.25 e1 and 0.75x + 0.375 + 0.375 e2, so the difference is
    0.25x + 0.125 + 0.375 e2 - 0.25 e1, at most 1.0.
    Naive intervals: [0, 1.5] - [0, 1] = [-1, 1.5], so 1.5."""
    a, b = _pair([1.0], [1.5], relu=True)
    bound = MaxBound(a, b, Box(torch.tensor([-1.0]), torch.tensor([1.0])), fp_model="real")
    expected = {"zonotope": 0.5, "interval": 0.5, "zonotope-separate": 1.0, "interval-separate": 1.5}
    for name, value in expected.items():
        assert bound.method_bounds[name] == pytest.approx(value, abs=1e-9)
    assert float(bound) == pytest.approx(0.5, abs=1e-9)


def _two_layer(w1, w2):
    """x (1 input) -> Linear(1, 2, no bias) -> ReLU -> Linear(2, 1, no bias)."""
    net = nn.Sequential(nn.Linear(1, 2, bias=False), nn.ReLU(), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(w1))
        net[2].weight.copy_(torch.tensor(w2))
    return net.eval()


@pytest.mark.parametrize("case", ["on-off", "off-on"])
def test_relu_switching_cases_are_exact(case):
    """Inputs x in [1, 2]. Neuron 2 is on in both networks (x versus 2x).
    on-off: neuron 1 is x in the original and -x (so off) in the compiled model.
      Original output x - x = 0, compiled 0 - 2x: difference -2x, largest 4.
    off-on: neuron 1 is -x (off) in the original and x in the compiled model.
      Original output 0 + x, compiled x + 2x: difference 2x, largest 4.
    The table in docs/DESIGN.md (section 5.2) makes both cases exact."""
    if case == "on-off":
        a, b = _two_layer([[1.0], [1.0]], [[1.0, -1.0]]), _two_layer([[-1.0], [2.0]], [[1.0, -1.0]])
    else:
        a, b = _two_layer([[-1.0], [1.0]], [[1.0, 1.0]]), _two_layer([[1.0], [2.0]], [[1.0, 1.0]])
    bound = MaxBound(a, b, Box(torch.tensor([1.0]), torch.tensor([2.0])), fp_model="real")
    assert float(bound) == pytest.approx(4.0, abs=1e-12)
    # the other analyses (README hand-case table): intervals lose the link between the two neurons
    expected = {"on-off": {"zonotope": 4, "interval": 4, "zonotope-separate": 4, "interval-separate": 5},
                "off-on": {"zonotope": 4, "interval": 5, "zonotope-separate": 4, "interval-separate": 5}}[case]
    for name, value in expected.items():
        assert bound.method_bounds[name] == pytest.approx(value, abs=1e-9), name


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid"])
def test_identical_models_get_exactly_zero(act):
    """Differential analysis returns exactly 0.0 for two copies of a model;
    analysing the copies separately cannot, because it forgets they are equal."""
    m = make_mlp([6, 16, 16, 3], act, seed=3)
    X = Box(-torch.ones(6), torch.ones(6))
    real = MaxBound(m, copy.deepcopy(m), X, fp_model="real")
    assert real.method_bounds["zonotope"] == 0.0
    assert real.method_bounds["interval"] == 0.0
    assert real.method_bounds["interval-separate"] > 0.0
    assert float(real) == 0.0
    with_rounding = MaxBound(m, copy.deepcopy(m), X)
    assert with_rounding.real_bound == 0.0
    assert 0.0 < float(with_rounding) < 1e-3
