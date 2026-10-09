"""Our own float64 arithmetic must never cut a bound short. These tests compare
against exact rational arithmetic (fractions.Fraction)."""
from fractions import Fraction

import pytest
import torch

from maxbound.errors import NumericalInconsistency
from maxbound.graph import AffineOp
from maxbound.intervals import Interval, add, intersect, mul, scale, sub
from maxbound.rounding import U32, check_range, deflate, down, gamma, inflate, up
from maxbound.zonotope import SymbolPool, Zonotope, affine


def F(x):
    return Fraction(float(x))


def random_interval(seed, n=300):
    g = torch.Generator().manual_seed(seed)
    mag = 10.0 ** torch.randint(-6, 6, (n,), generator=g).double()
    lo = torch.randn(n, generator=g, dtype=torch.float64) * mag
    return Interval(lo, lo + torch.rand(n, generator=g, dtype=torch.float64) * mag)


def test_up_and_down_move_one_float_and_keep_exact_zero():
    x = torch.tensor([1.0, -2.5, 1e-300, 0.0], dtype=torch.float64)
    assert bool((up(x)[:3] > x[:3]).all()) and bool((down(x)[:3] < x[:3]).all())
    assert float(up(x)[3]) == 0.0 and float(down(x)[3]) == 0.0


def test_gamma_is_larger_than_n_times_u():
    assert gamma(10, U32) > 10 * U32
    with pytest.raises(NumericalInconsistency):
        gamma(2 ** 23, U32)


@pytest.mark.parametrize("n", [1, 7, 100, 1000])
def test_inflate_and_deflate_enclose_exact_dot_products(n):
    g = torch.Generator().manual_seed(n)
    a = torch.rand(n, generator=g, dtype=torch.float64) * 10.0 ** torch.randint(-5, 5, (n,), generator=g)
    b = torch.rand(n, generator=g, dtype=torch.float64) * 10.0 ** torch.randint(-5, 5, (n,), generator=g)
    computed = (a * b).sum()
    exact = sum(F(x) * F(y) for x, y in zip(a.tolist(), b.tolist()))
    assert F(inflate(computed, n)) >= exact
    assert F(deflate(computed, n)) <= exact


def test_interval_operations_contain_the_exact_results():
    a, b = random_interval(1), random_interval(2)
    s, d, p, k = add(a, b), sub(a, b), mul(a, b), scale(a, 0.1)
    for i in range(a.lo.numel()):
        al, ah, bl, bh = F(a.lo[i]), F(a.hi[i]), F(b.lo[i]), F(b.hi[i])
        assert F(s.lo[i]) <= al + bl and F(s.hi[i]) >= ah + bh
        assert F(d.lo[i]) <= al - bh and F(d.hi[i]) >= ah - bl
        corners = [al * bl, al * bh, ah * bl, ah * bh]
        assert F(p.lo[i]) <= min(corners) and F(p.hi[i]) >= max(corners)
        kc = [F(0.1) * al, F(0.1) * ah]
        assert F(k.lo[i]) <= min(kc) and F(k.hi[i]) >= max(kc)


def test_affine_layer_encloses_the_exact_rational_result():
    """A float64 matrix product is off by a few units in the last place. The
    slack that zonotope.affine adds must cover it. We test on a single point, so
    no input width can hide a missing slack, against exact rational arithmetic."""
    g = torch.Generator().manual_seed(0)
    W = torch.randn(20, 200, generator=g, dtype=torch.float64)
    b = torch.randn(20, generator=g, dtype=torch.float64)
    x = 1000.0 * torch.randn(200, generator=g, dtype=torch.float64)
    op = AffineOp("linear", W, b, (200,), (20,))
    out = affine(Zonotope.from_box(Interval.point(x), SymbolPool()), op).bounds()
    for i in range(20):
        exact = sum(F(W[i, j]) * F(x[j]) for j in range(200)) + F(b[i])
        assert F(out.lo[i]) <= exact <= F(out.hi[i])


def test_center_and_radius_cover_the_interval():
    iv = random_interval(3)
    c, r = iv.center(), iv.radius()
    for i in range(c.numel()):
        assert F(c[i]) - F(r[i]) <= F(iv.lo[i]) and F(c[i]) + F(r[i]) >= F(iv.hi[i])


def test_disjoint_enclosures_raise_instead_of_returning_a_bound():
    a = Interval(torch.tensor([0.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    b = Interval(torch.tensor([2.0], dtype=torch.float64), torch.tensor([3.0], dtype=torch.float64))
    with pytest.raises(NumericalInconsistency):
        intersect(a, b)


def test_range_guard_rejects_tiny_and_infinite_values():
    with pytest.raises(NumericalInconsistency):
        check_range(torch.tensor([1e-200], dtype=torch.float64))
    with pytest.raises(NumericalInconsistency):
        check_range(torch.tensor([float("inf")], dtype=torch.float64))
    check_range(torch.tensor([0.0, 1e-30, 1e30], dtype=torch.float64))
