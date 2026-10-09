"""Every activation rule must contain the true values, checked on fine grids."""
import pytest
import torch

from maxbound.activations import deriv_range, diff_range, relaxation, value_range
from maxbound.intervals import Interval, add

SIGMA = {
    "relu": torch.relu,
    "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, 0.1),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}
SLOPE = {"relu": 0.0, "leaky_relu": 0.1, "tanh": 0.0, "sigmoid": 0.0}


def _random_interval(seed, n=50, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    lo = scale * torch.randn(n, generator=g, dtype=torch.float64)
    return Interval(lo, lo + scale * torch.rand(n, generator=g, dtype=torch.float64))


def _grid(iv, k=401):
    t = torch.linspace(0, 1, k, dtype=torch.float64)[:, None]
    return iv.lo + t * (iv.hi - iv.lo)


@pytest.mark.parametrize("kind", ["tanh", "sigmoid"])
def test_derivative_bounds_hold_on_a_fine_grid(kind):
    iv = _random_interval(1)
    x = _grid(iv).requires_grad_(True)
    (SIGMA[kind](x).sum()).backward()
    d = deriv_range(kind, iv)
    # autograd computes tanh' as 1 - tanh(x)^2, which loses ~1e-13 near saturation
    assert bool((x.grad >= d.lo - 1e-12).all() and (x.grad <= d.hi + 1e-12).all())


@pytest.mark.parametrize("kind", list(SIGMA))
def test_value_range_and_relaxation_contain_the_function(kind):
    iv = _random_interval(2)
    x = _grid(iv)
    y = SIGMA[kind](x)
    v = value_range(kind, SLOPE[kind], iv)
    assert bool((y >= v.lo).all() and (y <= v.hi).all())
    lam, mid, rad = relaxation(kind, SLOPE[kind], iv)
    crosses = (iv.lo < 0) & (iv.hi > 0)
    ok = (y - lam * x - mid).abs() <= rad + 1e-12
    if kind in ("relu", "leaky_relu"):
        ok = ok | ~crosses        # the ReLU relaxation is used only where the sign can change
    assert bool(ok.all())


@pytest.mark.parametrize("kind", list(SIGMA))
def test_difference_range_contains_sampled_values(kind):
    z = _random_interval(3)
    delta = _random_interval(4, scale=0.5)
    d = diff_range(kind, SLOPE[kind], z, delta, add(z, delta))
    t = torch.linspace(0, 1, 61, dtype=torch.float64)
    zz = (z.lo + t[:, None, None] * (z.hi - z.lo))
    dd = (delta.lo + t[None, :, None] * (delta.hi - delta.lo))
    f = SIGMA[kind](zz + dd) - SIGMA[kind](zz)
    assert bool((f >= d.lo - 1e-12).all() and (f <= d.hi + 1e-12).all())


def test_difference_of_exactly_zero_stays_zero():
    z = _random_interval(5)
    zero = Interval(torch.zeros(50, dtype=torch.float64), torch.zeros(50, dtype=torch.float64))
    for kind in SIGMA:
        d = diff_range(kind, SLOPE[kind], z, zero, z)
        assert float(d.lo.abs().max()) == 0.0 and float(d.hi.abs().max()) == 0.0
