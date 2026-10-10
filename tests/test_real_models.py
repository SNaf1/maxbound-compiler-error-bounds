"""The real Hugging Face model, torch.compile, and the experiment runner.

Tests marked ``network`` download dacorvo/mnist-mlp and the MNIST test split from
the Hugging Face Hub on first use (about 4 MB). Skip them with  -m "not network".
The torch.compile test needs a C++ compiler on PATH (on Windows: a Visual Studio
developer prompt) and is skipped otherwise.
"""
import shutil

import pytest
import torch

from maxbound import LinfBall, MaxBound
from maxbound.compilers import quantize_weights, torch_compile
from maxbound.empirical import attack_max_difference, sampled_max_difference, without_final_softmax
from maxbound.experiment import load_images, load_model, summarize, to_markdown

from conftest import make_mlp

MODEL = {"id": "dacorvo/mnist-mlp", "revision": "7ed1e96d182395fd7380ea8495b9eb2364ce9dc1", "trust_remote_code": True}
IMAGES = {"repo": "ylecun/mnist", "file": "mnist/test-00000-of-00001.parquet",
          "revision": "77f3279092a1c1579b2250db8eafed0ad422088c", "start": 0, "count": 3}
MEAN, STD = 0.1307, 0.3081


@pytest.mark.network
def test_hugging_face_model_with_8bit_weights():
    pytest.importorskip("transformers")
    model = load_model(MODEL)
    cl = quantize_weights(model, 8)
    pixels, labels = load_images(IMAGES)
    lo, hi = (float(v) for v in (torch.tensor([0.0, 1.0]) - MEAN) / STD)
    for x, label in zip(pixels, labels):
        X = LinfBall((x - MEAN) / STD, 0.003 / STD, clip=(lo, hi))
        bound = MaxBound(model, cl, X)
        seen = max(sampled_max_difference(model, cl, X, n=500)[0], attack_max_difference(model, cl, X, steps=30)[0])
        assert seen <= float(bound)
        assert bound.same_prediction == label           # the test images are classified correctly
        assert bound.method_bounds["zonotope"] <= bound.method_bounds["interval-separate"]


def _have_cpp_compiler():
    return any(shutil.which(c) for c in ("cl", "g++", "clang++"))


@pytest.mark.skipif(not _have_cpp_compiler(), reason="torch.compile needs a C++ compiler on PATH")
def test_torch_compile_difference_is_pure_rounding():
    """Inductor keeps the real-number function, so the real-arithmetic part is 0
    and the bound is the two rounding allowances; the compiled model's actual
    deviation must stay inside it."""
    m = make_mlp([16, 32, 32, 4], "relu", seed=0, softmax=True)
    cl = torch_compile(m)
    X = LinfBall(torch.randn(16, generator=torch.Generator().manual_seed(0)), 0.5)
    bound = MaxBound(m, cl, X)
    assert bound.real_bound == 0.0 and float(bound) > 0.0
    seen, _ = sampled_max_difference(m, cl, X, n=2000)
    assert seen <= float(bound)


def test_without_final_softmax_returns_the_logits():
    m = make_mlp([6, 8, 3], "relu", seed=1, softmax=True)
    bare = without_final_softmax(m)
    x = torch.randn(5, 6)
    assert torch.allclose(torch.softmax(bare(x), dim=-1), m(x))
    with pytest.raises(ValueError):
        without_final_softmax(bare)


def test_summary_counts_violations_and_certificates():
    rows = [dict(output="model", eps_pixels=0.01, bound=1.0, real=0.5, fp_allowance=0.5, found=0.4,
                 methods={"zonotope": 0.5}, certified_class=3, seconds=0.1, image=0, label=3),
            dict(output="model", eps_pixels=0.01, bound=1.0, real=0.5, fp_allowance=0.5, found=1.2,
                 methods={"zonotope": 0.5}, certified_class=None, seconds=0.1, image=1, label=1)]
    s = summarize(rows)[0]
    assert s["violations"] == 1 and s["certified"] == 1 and s["images"] == 2
    md = to_markdown({"config": {"model": {"id": "m"}, "compiler": {"name": "c"}}, "rows": rows, "summary": [s]})
    assert "| probabilities | 0.01 | 1/2 | 1/2 |" in md
