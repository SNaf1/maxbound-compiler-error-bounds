"""Run MaxBound from a JSON config: the data-driven entry point.

    python -m maxbound examples/configs/mnist_quant8.json --out results/mnist_quant8

A config names the model, the input images, the compiler, the input domain and
the MaxBound settings (see examples/configs/). For every image and every
radius eps it computes the bound and, as a sanity check, the largest difference
that sampling and a gradient attack can find. The bound must never be smaller
than what they find; the run counts and reports any such violation.

To try another model or dataset, change the config; to add another compile
pass, add it to COMPILERS.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn

from . import compilers
from .api import MaxBound
from .domain import LinfBall
from .empirical import attack_max_difference, sampled_max_difference, without_final_softmax

logger = logging.getLogger(__name__)

COMPILERS: Dict[str, Callable[..., nn.Module]] = {
    "quantize_weights": compilers.quantize_weights,
    "cast_weights": compilers.cast_weights,
    "prune_magnitude": compilers.prune_magnitude,
    "fuse_batchnorm": compilers.fuse_batchnorm,
    "torch_compile": compilers.torch_compile,
}


def load_model(spec: Dict[str, Any]) -> nn.Module:
    """A Hugging Face model, pinned to a revision. trust_remote_code runs the
    model's own Python file from the Hub; read that file before enabling it."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(spec["id"], revision=spec.get("revision"),
                                      trust_remote_code=spec.get("trust_remote_code", False))
    return model.eval()


def load_images(spec: Dict[str, Any]) -> Tuple[torch.Tensor, List[int]]:
    """Images from a Hugging Face dataset stored as parquet (like ylecun/mnist),
    as float32 pixels in [0, 1], flattened, plus their labels."""
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec.get("revision"))
    table = pq.read_table(path)
    start, count = spec.get("start", 0), spec["count"]
    images = table.column("image").to_pylist()[start:start + count]
    labels = table.column("label").to_pylist()[start:start + count]
    pixels = [np.asarray(Image.open(io.BytesIO(im["bytes"])), dtype=np.float32).reshape(-1) / 255.0 for im in images]
    return torch.from_numpy(np.stack(pixels)), labels


def make_compiler(spec: Dict[str, Any]) -> Callable[[nn.Module], nn.Module]:
    """Turn a config entry {"name": ..., "args": {...}} into a function model -> compiled model."""
    fn = COMPILERS[spec["name"]]
    kwargs = {k: (getattr(torch, v) if k == "dtype" else v) for k, v in spec.get("args", {}).items()}
    return lambda model: fn(model, **kwargs)


def run(config: Dict[str, Any], log: Callable[[str], None] = logger.info) -> Dict[str, Any]:
    """Run one experiment config (every image and radius: bound plus empirical check) and return rows and a summary."""
    model = load_model(config["model"])
    cl_func = make_compiler(config["compiler"])
    pixels, labels = load_images(config["inputs"])
    mean, std = config["inputs"]["normalize"]
    # the valid pixel range [0, 1] in normalised units, computed in float32 exactly like the images
    lo, hi = (float(v) for v in (torch.tensor([0.0, 1.0]) - mean) / std)
    checks = config.get("checks", {})
    pairs = {"model": (model, cl_func(model))}
    if config.get("also_logits", True):
        bare = without_final_softmax(model)
        pairs["logits"] = (bare, cl_func(bare))

    rows: List[Dict[str, Any]] = []
    for eps in config["domain"]["eps_pixels"]:
        for i in range(len(labels)):
            X = LinfBall((pixels[i] - mean) / std, eps / std, clip=(lo, hi))
            for output, (a, b) in pairs.items():
                t0 = time.perf_counter()
                bound = MaxBound(a, b, X, config.get("maxbound", {}))
                seconds = time.perf_counter() - t0
                sampled, _ = sampled_max_difference(a, b, X, n=checks.get("samples", 1000), seed=i)
                attacked, _ = attack_max_difference(a, b, X, steps=checks.get("attack_steps", 50),
                                                    restarts=checks.get("attack_restarts", 1), seed=i)
                found = max(sampled, attacked)
                rows.append(dict(eps_pixels=eps, image=config["inputs"].get("start", 0) + i, label=labels[i],
                                 output=output, bound=float(bound), real=bound.real_bound,
                                 fp_allowance=bound.fp_allowance, found=found, methods=bound.method_bounds,
                                 certified_class=bound.same_prediction, seconds=seconds))
            last = next(r for r in reversed(rows) if r["output"] == "model")
            log(f"eps={eps} image {i + 1}/{len(labels)}: bound {last['bound']:.3g} (found {last['found']:.3g})")
    return {"config": config, "environment": environment(), "rows": rows, "summary": summarize(rows)}


def environment() -> Dict[str, Any]:
    """What the results were produced with, so they can be reproduced."""
    import platform
    import sys
    from . import __version__
    return {"maxbound": __version__, "python": sys.version.split()[0], "torch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "threads": torch.get_num_threads()}


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Median statistics per (output, eps) over the images."""
    out = []
    keys = sorted({(r["output"], r["eps_pixels"]) for r in rows}, key=lambda k: (k[0] != "model", k[1]))
    for output, eps in keys:
        rs = [r for r in rows if r["output"] == output and r["eps_pixels"] == eps]
        med = lambda f: statistics.median(f(r) for r in rs)
        out.append(dict(
            output="probabilities" if output == "model" else "logits", eps_pixels=eps, images=len(rs),
            violations=sum(r["found"] > r["bound"] for r in rs),
            certified=sum(r["certified_class"] is not None for r in rs),
            bound=med(lambda r: r["bound"]), real=med(lambda r: r["real"]),
            fp_allowance=med(lambda r: r["fp_allowance"]), found=med(lambda r: r["found"]),
            real_over_found=med(lambda r: r["real"] / r["found"] if r["found"] > 0 else float("inf")),
            methods={m: statistics.median(r["methods"][m] for r in rs) for m in rs[0]["methods"]},
            seconds=med(lambda r: r["seconds"]),
        ))
    return out


def to_markdown(results: Dict[str, Any]) -> str:
    """The results as a Markdown table."""
    cfg = results["config"]
    lines = [f"# {cfg.get('name', 'MaxBound run')}", "",
             f"Model `{cfg['model']['id']}` @ `{cfg['model'].get('revision', 'latest')}`, "
             f"compiler `{cfg['compiler']['name']}` {cfg['compiler'].get('args', {})}, "
             f"{len({r['image'] for r in results['rows']})} test images. "
             "Medians over images; eps is in pixel units (pixels in [0, 1]).", "",
             "| output | eps | violations | same class certified | bound | real part | fp allowance | "
             "largest found | real / found | zonotope | interval | zonotope-sep. | interval-sep. | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in results["summary"]:
        m = s["methods"]
        cols = [m.get(k) for k in ("zonotope", "interval", "zonotope-separate", "interval-separate")]
        lines.append("| " + " | ".join([
            s["output"], f"{s['eps_pixels']:g}", f"{s['violations']}/{s['images']}", f"{s['certified']}/{s['images']}",
            f"{s['bound']:.3g}", f"{s['real']:.3g}", f"{s['fp_allowance']:.3g}", f"{s['found']:.3g}",
            f"{s['real_over_found']:.2f}", *[f"{c:.3g}" if c is not None else "n/a" for c in cols],
            f"{s['seconds']:.2f}"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: List[str] = None) -> None:
    """Command-line entry point: python -m maxbound <config.json> [--out PATH]."""
    parser = argparse.ArgumentParser(prog="python -m maxbound", description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="path to a JSON config")
    parser.add_argument("--out", help="write <out>.json and <out>.md")
    parser.add_argument("--verbose", action="store_true", help="also log every MaxBound call step by step")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    if not args.verbose:
        logging.getLogger("maxbound.api").setLevel(logging.WARNING)   # one progress line per image instead
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results = run(config)
    print(to_markdown(results))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        out.with_suffix(".md").write_text(to_markdown(results), encoding="utf-8")
