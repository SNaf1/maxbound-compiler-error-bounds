"""Where the true maximum can be computed independently, the zonotope analysis
must reproduce it."""
import copy

import pytest
import torch

from maxbound import Box, MaxBound

from conftest import make_mlp, perturb


def test_bound_equals_true_maximum_when_no_relu_can_switch():
    """Positive weights, positive biases and positive inputs keep every ReLU on,
    so both networks are affine on X: y = A x + c. The largest difference is then
    |dA center + dc| + |dA| radius, which we compute directly with matrices."""
    m = make_mlp([5, 8, 4], "relu", seed=0)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.abs() + 0.05)
    cl = copy.deepcopy(m)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in cl.parameters():
            p.mul_(1.0 + 0.05 * torch.rand(p.shape, generator=g))   # stays positive
    lower, upper = torch.full((5,), 0.1), torch.ones(5)

    def affine_form(net):
        W1, b1 = net[0].weight.detach().double(), net[0].bias.detach().double()
        W2, b2 = net[2].weight.detach().double(), net[2].bias.detach().double()
        return W2 @ W1, W2 @ b1 + b2

    (A, c), (A2, c2) = affine_form(m), affine_form(cl)
    center, radius = (lower.double() + upper.double()) / 2, (upper.double() - lower.double()) / 2
    exact = ((A2 - A) @ center + (c2 - c)).abs() + (A2 - A).abs() @ radius

    bound = MaxBound(m, cl, Box(lower, upper), fp_model="real")
    assert bound.method_bounds["zonotope"] == pytest.approx(float(exact.max()), rel=1e-9)
    assert bound.method_bounds["interval"] >= bound.method_bounds["zonotope"]
    assert bound.relaxed_neurons["zonotope"] == 0


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid"])
def test_zero_width_box_reproduces_the_forward_pass(act):
    """On a single point every analysis must return the actual difference of the
    two networks at that point (computed here in float64). This checks graph
    extraction and every layer rule at once."""
    m = make_mlp([6, 12, 12, 4], act, seed=7)
    cl = perturb(m, seed=8)
    x = torch.randn(6)
    bound = MaxBound(m, cl, Box(x, x), fp_model="real")
    with torch.no_grad():
        true = float((copy.deepcopy(cl).double()(x.double()) - copy.deepcopy(m).double()(x.double())).abs().max())
    for name, value in bound.method_bounds.items():
        assert value == pytest.approx(true, rel=1e-9, abs=1e-12), name
