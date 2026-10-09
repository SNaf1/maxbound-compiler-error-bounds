"""Model extraction: what is supported, what is rejected, and BatchNorm folding."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxbound import Box, MaxBound, UnsupportedModelError
from maxbound.compilers import fuse_batchnorm
from maxbound.empirical import sampled_max_difference
from maxbound.graph import ActOp, AffineOp, ReshapeOp, SoftmaxOp, extract

from conftest import make_mlp, randomize_batchnorm


class Functional(nn.Module):
    """Written with torch.nn.functional calls and x.view(x.size(0), -1), the
    style many Hugging Face model files use."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 12))
        self.b1 = nn.Parameter(torch.randn(16))
        self.w2 = nn.Parameter(torch.randn(3, 16))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, self.w1, self.b1))
        return F.softmax(F.linear(x, self.w2), dim=-1)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nn.Linear(4, 4), nn.Linear(4, 4)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x))) + x


def test_sequential_mlp_is_extracted():
    chain = extract(make_mlp([4, 8, 3], "tanh", softmax=True), torch.zeros(4))
    assert [type(op) for op in chain.ops] == [AffineOp, ActOp, AffineOp, SoftmaxOp]
    assert chain.in_shape == (4,) and chain.out_shape == (3,) and chain.dtype == torch.float32


def test_functional_model_is_extracted():
    chain = extract(Functional().eval(), torch.zeros(3, 4))
    assert [type(op) for op in chain.ops] == [ReshapeOp, AffineOp, ActOp, AffineOp, SoftmaxOp]
    assert not chain.ops[3].has_bias


@pytest.mark.parametrize("model", [Residual(), nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))])
def test_unsupported_structures_are_rejected(model):
    with pytest.raises(UnsupportedModelError):
        extract(model.eval(), torch.zeros(4))


def test_training_mode_is_rejected():
    with pytest.raises(ValueError):
        extract(make_mlp([4, 3]).train(), torch.zeros(4))


def test_models_that_do_not_line_up_fall_back_to_separate_analysis():
    m, other = make_mlp([4, 8, 2], seed=0), make_mlp([4, 6, 2], seed=1)
    X = Box(-torch.ones(4), torch.ones(4))
    bound = MaxBound(m, other, X)
    assert bound.alignment_problem is not None
    assert set(bound.method_bounds) == {"zonotope-separate", "interval-separate"}
    seen, _ = sampled_max_difference(m, other, X, n=2000)
    assert seen <= float(bound)


def test_batchnorm_fusion_is_verified_not_assumed():
    """Linear -> BatchNorm in the original, one fused Linear in the compiled model.
    Folding makes the chains line up; the bound then covers the float32
    rounding PyTorch introduced while computing the fused weights."""
    m = randomize_batchnorm(nn.Sequential(nn.Linear(6, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10, 3)), 2)
    fused = fuse_batchnorm(m)
    X = Box(-torch.ones(6), torch.ones(6))
    bound = MaxBound(m, fused, X)
    assert bound.alignment_problem is None
    assert bound.real_bound < 1e-4
    seen, _ = sampled_max_difference(m, fused, X, n=2000)
    assert seen <= float(bound)
