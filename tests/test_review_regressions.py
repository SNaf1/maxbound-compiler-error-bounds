"""Counterexamples found in an independent review (each one broke an earlier
version of the package). They stay here so the bugs cannot come back."""
import copy
from decimal import Decimal, getcontext

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from maxbound import Box, LinfBall, MaxBound, NumericalInconsistency, UnsupportedModelError
from maxbound.activations import deriv_range
from maxbound.intervals import Interval

getcontext().prec = 70


@torch.no_grad()
def observed(a, b, x):
    return (a(x[None]).double() - b(x[None]).double()).abs().max().item()


@pytest.mark.parametrize("flush", [False, True])
def test_batchnorm_with_a_subnormal_scale(flush):
    """weight 1e-25 / sqrt(1e38) = 1e-44 is subnormal in float32: the scale loses
    most of its digits (or becomes 0 under flush-to-zero), and that absolute error
    is then multiplied by x = 1e30."""
    if not torch.set_flush_denormal(flush):
        if flush:
            pytest.skip("flush-to-zero is not supported on this CPU")
    try:
        _batchnorm_with_a_subnormal_scale()
    finally:
        torch.set_flush_denormal(False)


def _batchnorm_with_a_subnormal_scale():
    model = nn.Sequential(nn.BatchNorm1d(1, eps=0.0)).eval()
    cl = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1e-25)
        model[0].bias.zero_()
        model[0].running_mean.zero_()
        model[0].running_var.fill_(1e38)
        cl[0].weight.fill_(1e-19)
        cl[1].weight.copy_(model[0].weight.reshape(1, 1))
    x = torch.tensor([1e30])
    assert observed(model, cl, x) <= float(MaxBound(model, cl, Box(x, x)))


def _bn_against_linear(eps, var, w_cl):
    model = nn.Sequential(nn.BatchNorm1d(1, eps=eps)).eval()
    cl = nn.Sequential(nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].running_var.fill_(var)
        model[0].running_mean.zero_()
        model[0].bias.zero_()
        model[0].weight.fill_(1.0)
        cl[0].weight.fill_(w_cl)
    return model, cl


def test_batchnorm_eps_that_underflows_in_float32_is_refused():
    """eps = 1e-44 becomes 9.8e-45 in float32 (and 0 under flush-to-zero), so with
    var = 0 the executed scale is off by 1% or infinite. We refuse the model."""
    model, cl = _bn_against_linear(1e-44, 0.0, 1e22)
    x = torch.tensor([1e-22])
    with pytest.raises(NumericalInconsistency):
        MaxBound(model, cl, Box(x, x))


def test_negative_running_variance_is_refused():
    model, cl = _bn_against_linear(1e-5, -0.99999e-5, 1.0)
    with pytest.raises(UnsupportedModelError):
        MaxBound(model, cl, Box(torch.ones(1), torch.ones(1)))


@pytest.mark.parametrize("eps,var", [(1e-5, 0.0), (1e-30, 0.0), (1e-5, 1e-6), (0.1, 3.0)])
def test_batchnorm_with_small_denominators_stays_sound(eps, var):
    model, cl = _bn_against_linear(eps, var, 1.0)
    X = LinfBall(torch.zeros(1), 5.0)
    xs = X.to_box().sample(500, seed=0)
    with torch.no_grad():
        seen = (model(xs[:, None].reshape(-1, 1)).double() - cl(xs.reshape(-1, 1)).double()).abs().max().item()
    assert seen <= float(MaxBound(model, cl, X))


def test_live_inductor_fast_math_is_refused():
    from torch._inductor import config as inductor_config
    from maxbound.compilers import torch_compile
    m = nn.Sequential(nn.Linear(2, 2)).eval()
    old = getattr(inductor_config, "use_fast_math", False)
    inductor_config.use_fast_math = True
    try:
        with pytest.raises(RuntimeError):
            MaxBound(m, torch_compile(m), Box(torch.zeros(2), torch.ones(2)))
    finally:
        inductor_config.use_fast_math = old


def test_leaky_relu_slope_is_rounded_to_float32_before_use():
    model = nn.Sequential(nn.LeakyReLU(0.9999999105930327)).eval()
    cl = nn.Sequential(nn.Identity()).eval()
    x = torch.tensor([-1.5])
    assert observed(model, cl, x) <= float(MaxBound(model, cl, Box(x, x)))


def test_list_domains_are_kept_in_float64():
    """1.00000004 is not a float32 value; converting it to float32 first used to
    move the ball and exclude valid inputs."""
    model = nn.Sequential(nn.Identity()).eval()
    cl = nn.Sequential(nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        cl[0].weight.zero_()
    X = LinfBall([1.00000004], 1e-7)
    x = torch.tensor([1.0000001192092896])
    assert abs(float(x[0]) - 1.00000004) <= 1e-7
    assert observed(model, cl, x) <= float(MaxBound(model, cl, X))


def test_certificate_is_refused_when_probabilities_can_tie():
    """Logits [0, 1e-8] and [0, 1.1e-7] are strictly ordered, but rounded softmax
    gives [0.5, 0.5] for the first, so the two models' argmax can differ."""
    model = nn.Sequential(nn.Softmax(dim=-1)).eval()
    cl = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1)).eval()
    with torch.no_grad():
        cl[0].weight.copy_(torch.eye(2))
        cl[0].bias.copy_(torch.tensor([0.0, 1e-7]))
    x = torch.tensor([0.0, 1e-8])
    assert MaxBound(model, cl, Box(x, x)).same_prediction is None


@pytest.mark.parametrize("x", [0.0, 3.0, 20.0, 100.0, 356.0, 400.0, 800.0])
def test_tanh_and_sigmoid_slopes_far_in_the_tail(x):
    ref_tanh = 4 / ((Decimal(x).exp() + (-Decimal(x)).exp()) ** 2)
    e = (-Decimal(x)).exp()
    ref_sigmoid = e / (1 + e) ** 2
    point = Interval(torch.tensor([x], dtype=torch.float64), torch.tensor([x], dtype=torch.float64))
    for kind, ref in (("tanh", ref_tanh), ("sigmoid", ref_sigmoid)):
        d = deriv_range(kind, point)
        assert Decimal(d.lo.item()) <= ref <= Decimal(d.hi.item()), kind


def test_reduced_precision_convolution_setting_is_refused():
    m = nn.Sequential(nn.Linear(2, 2)).eval()
    torch.backends.mkldnn.conv.fp32_precision = "bf16"
    try:
        with pytest.raises(RuntimeError):
            MaxBound(m, m, Box(torch.zeros(2), torch.ones(2)))
    finally:
        torch.backends.mkldnn.conv.fp32_precision = "none"


class HalfSoftmax(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1, dtype=torch.float16)


def test_operations_that_change_dtype_are_refused():
    x = torch.tensor([0.1, 0.2])
    with pytest.raises(UnsupportedModelError):
        MaxBound(HalfSoftmax().eval(), nn.Sequential(nn.Softmax(dim=-1)).eval(), Box(x, x))


def test_float64_models_accept_points_that_are_not_float32_values():
    m = nn.Sequential(nn.Linear(1, 1)).double().eval()
    x = torch.tensor([0.1], dtype=torch.float64)
    bound = MaxBound(m, copy.deepcopy(m), Box(x, x))
    assert bound.real_bound == 0.0 and float(bound) < 1e-12
