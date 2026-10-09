"""The example compile passes do what they claim and leave the original alone."""
import copy

import torch
import torch.nn as nn

from maxbound.compilers import cast_weights, fuse_batchnorm, prune_magnitude, quantize_weights

from conftest import make_mlp, randomize_batchnorm


def test_quantize_weights_rounds_to_the_grid():
    m = make_mlp([10, 20, 5], seed=0)
    before = copy.deepcopy(m)
    q = quantize_weights(m, bits=8)
    for (name, p), p0, pq in zip(m.named_parameters(), before.parameters(), q.parameters()):
        assert torch.equal(p, p0)                        # original untouched
        if p.dim() < 2:
            assert torch.equal(pq, p)                    # biases unchanged
            continue
        scale = p.abs().amax(dim=1, keepdim=True) / 127
        assert bool(((pq - p).abs() <= scale / 2 + 1e-7).all())
        steps = pq / scale
        assert bool(((steps - steps.round()).abs() < 1e-3).all())


def test_prune_magnitude_reaches_the_sparsity():
    q = prune_magnitude(make_mlp([10, 20, 5], seed=1), 0.5)
    w = q[0].weight
    assert abs(float((w == 0).float().mean()) - 0.5) < 0.01


def test_cast_weights_round_trips_through_half_precision():
    q = cast_weights(make_mlp([10, 20, 5], seed=2), torch.float16)
    for p in q.parameters():
        assert p.dtype == torch.float32 and torch.equal(p, p.half().float())


def test_fuse_batchnorm_keeps_the_function():
    m = randomize_batchnorm(nn.Sequential(nn.Linear(6, 10), nn.BatchNorm1d(10), nn.ReLU(), nn.Linear(10, 3)), 3)
    fused = fuse_batchnorm(m)
    assert not any(isinstance(mod, nn.BatchNorm1d) for mod in fused.modules())
    x = torch.randn(50, 6)
    assert torch.allclose(m(x), fused(x), atol=1e-5)
