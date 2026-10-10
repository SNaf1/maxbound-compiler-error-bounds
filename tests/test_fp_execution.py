"""The floating-point allowance and the assumptions behind it."""
import copy
from decimal import Decimal, getcontext

import pytest
import torch
import torch.nn as nn

from maxbound import LinfBall
from maxbound.activations import LIBM_REL
from maxbound.fp_execution import TRANS_U, execution_allowance
from maxbound.graph import extract
from maxbound.rounding import U32

from conftest import make_mlp, randomize_batchnorm, small_cnn

getcontext().prec = 50


def _exact(name, x):
    x = Decimal(x)
    if name == "exp":
        return x.exp()
    if name == "tanh":
        e = (2 * x).exp()
        return (e - 1) / (e + 1)
    return 1 / (1 + (-x).exp())


def _max_rel_error(name, fn, xs):
    worst = Decimal(0)
    for x, v in zip(xs.tolist(), fn(xs).tolist()):
        ref = _exact(name, x)
        if ref != 0:
            worst = max(worst, abs((Decimal(v) - ref) / ref))
    return worst


@pytest.mark.parametrize("name,fn,lo,hi", [("exp", torch.exp, -700, 700), ("tanh", torch.tanh, -20, 20),
                                         ("sigmoid", torch.sigmoid, -700, 700)])
def test_float64_library_functions_meet_the_assumption(name, fn, lo, hi):
    xs = torch.cat([torch.linspace(lo, hi, 3001, dtype=torch.float64),
                    torch.randn(1000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)])
    assert _max_rel_error(name, fn, xs) <= Decimal(LIBM_REL)


@pytest.mark.parametrize("name,fn,lo,hi", [("exp", torch.exp, -80, 80), ("tanh", torch.tanh, -20, 20),
                                         ("sigmoid", torch.sigmoid, -80, 80)])
def test_float32_library_functions_meet_the_assumption(name, fn, lo, hi):
    xs = torch.cat([torch.linspace(lo, hi, 20001, dtype=torch.float32),
                    torch.randn(5000, generator=torch.Generator().manual_seed(0), dtype=torch.float32)])
    assert _max_rel_error(name, fn, xs) <= Decimal(TRANS_U * U32)


def _check_allowance(model, X, n=3000):
    box = X.to_box()
    chain = extract(model, box.center_float32())
    allowance = execution_allowance(chain.ops, box.interval(), torch.float32)
    pts = box.sample(n, seed=3)
    with torch.no_grad():
        y32 = model(pts).double().flatten(1)
        y64 = copy.deepcopy(model).double()(pts.double()).flatten(1)
    err = (y32 - y64).abs().max(0).values
    assert bool((err <= allowance.output).all())
    return float(err.max()), float(allowance.output.max())


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid"])
@pytest.mark.parametrize("softmax", [False, True])
def test_allowance_covers_observed_float32_rounding(act, softmax):
    m = make_mlp([10, 64, 64, 4], act, seed=5, softmax=softmax)
    X = LinfBall(torch.randn(10, generator=torch.Generator().manual_seed(6)), 1.0)
    observed, allowed = _check_allowance(m, X)
    assert 0.0 < observed < allowed


def test_allowance_covers_softmax_rounding_on_its_own():
    """A model that is only a softmax: no error arrives from earlier layers, so
    the allowance must come entirely from softmax's own rounding."""
    model = nn.Sequential(nn.Softmax(dim=-1)).eval()
    observed, allowed = _check_allowance(model, LinfBall(torch.randn(10, generator=torch.Generator().manual_seed(9)), 3.0))
    assert 0.0 < observed < allowed


def test_allowance_covers_batchnorm_and_convolutions():
    mlp = randomize_batchnorm(nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 3)), 1)
    _check_allowance(mlp, LinfBall(torch.zeros(10), 2.0))
    cnn = small_cnn(seed=3)
    _check_allowance(cnn, LinfBall(torch.full((1, 8, 8), 0.5), 0.5))
