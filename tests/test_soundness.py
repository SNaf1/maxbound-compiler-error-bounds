"""Sampling and a gradient attack must never beat the bound. They can only refute
a bound, never confirm one, so these tests are sanity checks: the proof is the
derivation in each module."""
import copy

import pytest
import torch

from maxbound import LinfBall, MaxBound
from maxbound.analysis import METHODS
from maxbound.compilers import fuse_batchnorm, prune_magnitude, quantize_weights
from maxbound.empirical import attack_max_difference, differences, sampled_max_difference

from conftest import make_mlp, perturb, small_cnn


def _ball(dim, seed, eps):
    g = torch.Generator().manual_seed(seed)
    return LinfBall(torch.randn(dim, generator=g), eps)


def _observed(model, cl, X):
    s, _ = sampled_max_difference(model, cl, X, n=2000, seed=1)
    a, _ = attack_max_difference(model, cl, X, steps=40, restarts=1)
    return max(s, a)


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid"])
@pytest.mark.parametrize("change", ["quantize4", "noise"])
@pytest.mark.parametrize("softmax", [False, True])
def test_no_sample_or_attack_beats_any_method(act, change, softmax):
    seed = sum(map(ord, act + change)) + int(softmax)
    m = make_mlp([8, 20, 20, 5], act, seed=seed, softmax=softmax)
    cl = quantize_weights(m, bits=4) if change == "quantize4" else perturb(m, seed=seed + 1)
    X = _ball(8, seed, 0.3)
    seen = _observed(m, cl, X)
    best = MaxBound(m, cl, X)
    assert seen <= float(best)
    for name in METHODS:
        assert seen <= float(MaxBound(m, cl, X, method=name))


@pytest.mark.parametrize("act", ["relu", "tanh"])
def test_float64_models_get_a_bound_for_float64_execution(act):
    """With float64 models the rounding allowance is tiny, so the bound is close
    to the real-arithmetic bound and still never beaten."""
    m = make_mlp([8, 20, 20, 5], act, seed=11)
    cl = quantize_weights(m, bits=4)
    m64, cl64 = copy.deepcopy(m).double(), copy.deepcopy(cl).double()
    X = _ball(8, 5, 0.3)
    bound = MaxBound(m64, cl64, X)
    assert bound.fp_allowance < 1e-9
    pts = X.to_box().sample(4000, seed=2).double()
    assert float(differences(m64, cl64, pts).max()) <= float(bound)


@pytest.mark.parametrize("cl_func", [lambda m: quantize_weights(m, bits=4), fuse_batchnorm,
                                     lambda m: prune_magnitude(m, 0.3)])
def test_convolutional_model(cl_func):
    m = small_cnn(seed=2)
    cl = cl_func(m)
    g = torch.Generator().manual_seed(0)
    X = LinfBall(torch.rand((1, 8, 8), generator=g), 0.1, clip=(0.0, 1.0))
    bound = MaxBound(m, cl, X)
    assert bound.alignment_problem is None
    assert _observed(m, cl, X) <= float(bound)


def test_bound_grows_with_the_box_for_interval_analysis():
    """Interval arithmetic is inclusion-monotone: a bigger box can only give a
    bigger (or equal) bound. (Zonotope relaxations do not promise this.)"""
    m = make_mlp([8, 20, 20, 5], "relu", seed=4)
    cl = quantize_weights(m, bits=6)
    center = torch.randn(8)
    values = [float(MaxBound(m, cl, LinfBall(center, eps), method="interval")) for eps in (0.0, 0.05, 0.1, 0.2, 0.4)]
    assert values == sorted(values)
