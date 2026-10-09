"""Planted-bug check (mutation testing) for maxbound's soundness.

A bound that is too small looks exactly like a good bound, so a soundness bug
can hide behind a passing test suite. This script plants one known bug at a time
into a copy of the package and runs the test suite against the copy. A bug is
"caught" if at least one test fails. Every bug below makes the bound unsound
(or breaks a stated guarantee), so every one of them should be caught.

    python tools/mutation_check.py            # from the package root
    python tools/mutation_check.py --out results/mutation_check.md
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "maxbound"

# (name, what the bug does, file, original text, planted text)
MUTANTS = [
    ("drop-weight-change", "difference at a linear layer forgets (W' - W) h + (b' - b)", "analysis.py",
     "delta = zt.add(zt.affine(d, op_b, bias=False), zt.affine(h, op_a.difference(op_b)))",
     "delta = zt.affine(d, op_b, bias=False)"),
    ("switching-relu-as-on", "a ReLU that can switch inside X is treated as always on", "analysis.py",
     "on, off = iz.lo >= 0, iz.hi <= 0", "on, off = iz.hi >= 0, iz.hi <= 0"),
    ("on-off-sign", "wrong sign when the neuron is on in the original and off in the compiled model",
     "analysis.py", "f_on_off = zt.sub(zt.scale(zp, a), z)", "f_on_off = zt.sub(z, zt.scale(zp, a))"),
    ("off-on-formula", "wrong formula when the neuron is off in the original and on in the compiled model",
     "analysis.py", "f_off_on = zt.sub(zp, zt.scale(z, a))", "f_off_on = zt.sub(z, zt.scale(z, a))"),
    ("wrong-corners", "four-corner rule takes the minimum at the wrong corners", "activations.py",
     "box = Interval(torch.minimum(f_lo(z.lo, delta.lo), f_lo(z.hi, delta.lo)),",
     "box = Interval(torch.minimum(f_lo(z.lo, delta.hi), f_lo(z.hi, delta.hi)),"),
    ("thin-relu-band", "ReLU relaxation band half as wide as needed", "activations.py",
     "half = 0.5 * gmax", "half = 0.25 * gmax"),
    ("deriv-at-midpoint", "largest tanh/sigmoid slope taken at the midpoint, not the point closest to 0",
     "activations.py", "top = _deriv(kind, x0)", "top = _deriv(kind, 0.5 * (iv.lo + iv.hi))"),
    ("forget-interval-part", "concretization ignores the interval part e of a form", "zonotope.py",
     "return inflate(self.G.abs().sum(0) + self.e, self.rows + 2)",
     "return inflate(self.G.abs().sum(0), self.rows + 2)"),
    ("no-float64-slack", "our own float64 rounding in linear layers is ignored", "zonotope.py",
     "err = abs_e + gamma(n + 1, U64) * terms", "err = abs_e"),
    ("no-outward-rounding", "endpoints are not rounded outward", "rounding.py",
     "return torch.where(x == 0, x, torch.nextafter(x, torch.full_like(x, math.inf)))", "return x"),
    ("softmax-half", "softmax bound off by a factor of 2", "softmax.py",
     "return up(max_pq(p) * spread)", "return up(max_pq(p) * spread * 0.5)"),
    ("bn-fold-no-shift", "BatchNorm folding drops the shift t", "graph.py",
     "b = prev.bias * sc.weight + sc.bias", "b = prev.bias * sc.weight"),
    ("no-dot-rounding", "float32 rounding of the models' own dot products is ignored", "fp_execution.py",
     "own = gamma(n, u) * size + n * tiny", "own = torch.zeros_like(size)"),
    ("no-softmax-rounding", "float32 rounding inside softmax is ignored", "fp_execution.py",
     "own = up(p.hi * kappa * 1.03) + tiny", "own = torch.zeros_like(p.hi)"),
    ("bn-no-underflow", "underflow of BatchNorm's precomputed scale is ignored", "fp_execution.py",
     "+ tiny * (mag_in + mean + 3.0))", "+ tiny)"),
    ("leaky-slope-exact", "LeakyReLU's slope is assumed to be stored exactly in float32", "fp_execution.py",
     "own = (u * stored + abs(stored - op.slope) * (1 + 4 * 2.0 ** -53)) * in_mag + tiny",
     "own = u * stored * in_mag + tiny"),
    ("domain-via-float32", "Python lists are converted to float32 before float64", "domain.py",
     "return torch.as_tensor(x, dtype=torch.float64).detach().clone()",
     "return torch.as_tensor(x).detach().to(torch.float64).clone()"),
    ("certificate-no-allowance", "the softmax certificate ignores the rounding allowance", "api.py",
     "ok &= ((low[:, None] > high[None, :]) | eye).all(1)",
     "ok &= ((p.lo[:, None] > p.hi[None, :]) | eye).all(1)"),
]


def run_tests(pythonpath: Path) -> tuple:
    env = dict(os.environ, PYTHONPATH=str(pythonpath))
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-m", "not network", "-p", "no:cacheprovider", "tests"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("FAILED")]
    return proc.returncode, (lines[0][7:] if lines else ""), time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write a markdown report here")
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "maxbound"
        # sanity: the unmodified copy must pass, and must be the one imported
        shutil.copytree(SRC, copy)
        probe = subprocess.run([sys.executable, "-c", "import maxbound; print(maxbound.__file__)"],
                               env=dict(os.environ, PYTHONPATH=tmp), capture_output=True, text=True)
        if not probe.stdout.strip().startswith(str(copy)):
            raise SystemExit(f"the copy is not the package being imported: {probe.stdout or probe.stderr}")
        code, _, secs = run_tests(Path(tmp))
        if code != 0:
            raise SystemExit("the unmodified package does not pass its tests")
        print(f"baseline: all tests pass ({secs:.0f} s)")
        for name, what, fname, old, new in MUTANTS:
            target = copy / fname
            original = (SRC / fname).read_text(encoding="utf-8")
            if original.count(old) != 1:
                raise SystemExit(f"{name}: expected exactly one occurrence of the original text in {fname}")
            target.write_text(original.replace(old, new), encoding="utf-8")
            code, failed, secs = run_tests(Path(tmp))
            target.write_text(original, encoding="utf-8")
            caught = code != 0
            rows.append((name, what, caught, failed))
            print(f"{'CAUGHT ' if caught else 'MISSED '} {name:22} {secs:5.0f} s  {failed}")
    caught = sum(r[2] for r in rows)
    print(f"\n{caught}/{len(rows)} planted bugs caught")
    if args.out:
        md = ["# Planted-bug check", "",
              f"{caught} of {len(rows)} planted soundness bugs are caught by the test suite "
              "(`python tools/mutation_check.py`).", "",
              "| bug | what it does | caught | first failing test |", "|---|---|---|---|"]
        md += [f"| `{n}` | {w} | {'yes' if c else '**no**'} | {f.split(' - ')[0] if f else ''} |" for n, w, c, f in rows]
        Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
