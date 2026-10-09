"""Input domains: conversion to boxes, sampling, configuration."""
import pytest
import torch

from maxbound import Box, LinfBall, as_box, domain_from_config


def test_linf_ball_contains_the_ball_and_respects_the_clip():
    c = torch.tensor([0.0, 0.5, 1.0])
    box = LinfBall(c, 0.1, clip=(0.0, 1.0)).to_box()
    assert bool((box.lower <= torch.clamp(c.double() - 0.1, min=0.0)).all())
    assert bool((box.upper >= torch.clamp(c.double() + 0.1, max=1.0)).all())
    assert float(box.lower.min()) >= 0.0 and float(box.upper.max()) <= 1.0


def test_samples_are_float32_points_inside_the_box():
    box = Box(torch.tensor([0.1, -3.0]), torch.tensor([0.1000001, 2.0]))
    pts = box.sample(1000, seed=0)
    assert pts.dtype == torch.float32
    assert all(box.contains(p) for p in pts)


def test_config_dicts_build_domains():
    ball = domain_from_config({"type": "linf_ball", "center": [0.0, 1.0], "eps": 0.5, "clip": [0, 1]})
    assert isinstance(ball, LinfBall)
    box = as_box({"type": "box", "lower": [0, 0], "upper": [1, 2]})
    assert box.upper.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        domain_from_config({"type": "sphere"})


def test_invalid_boxes_are_rejected():
    with pytest.raises(ValueError):
        Box(torch.tensor([1.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError):
        Box(torch.tensor([0.0]), torch.tensor([float("inf")]))
