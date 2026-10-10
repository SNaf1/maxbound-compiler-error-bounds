# maxbound

Sound upper bounds on how much a neural network's output can change when the network is compiled
(quantized, fused, or lowered by `torch.compile`), for every input in a region.

```python
import torch
from maxbound import MaxBound, LinfBall
from maxbound.compilers import quantize_weights

model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).eval()
cl_func = lambda m: quantize_weights(m, bits=8)      # the "compiler"
X = LinfBall(torch.zeros(4), eps=0.1)                # every x with |x_i| <= 0.1

cl_model = cl_func(model)
bound = MaxBound(model, cl_model, X)
print(float(bound))       # >= max over x in X of || model(x) - cl_model(x) ||_inf
print(bound.summary())
```

`examples/quickstart.py` runs the same two lines on a Hugging Face model. To see each step of the analysis,
turn on logging with `logging.basicConfig(level=logging.INFO)`; MaxBound reports to the `maxbound` logger.

**Results in brief** (Hugging Face MNIST classifier, 8-bit weights, 50 test images, radii up to 0.01 in pixel
units; details in [Results](#results)):

* The bound has two parts that are reported separately:
  * the real-arithmetic part (the difference if the computer calculated exactly);
  * a worst-case allowance for float32 rounding.
* **On the logits, the real-arithmetic part is tight:** a median of 1.00x to 2.12x the largest difference that
  sampling plus a gradient attack can find.
* **The full bound is looser:** a median of about 8x on the logits (4x to 17x across images) and about 85x to
  150x on the probabilities. The allowance assumes every rounding error goes the worst way, and it dominates.
* Bounding each model separately with intervals is about 630x looser than the real-arithmetic part at the
  largest radius.
* In 600 checks (8-bit quantization and `torch.compile`), no sample or attack exceeded the bound.

**Scope:** models that are a single chain of Linear, Conv2d, BatchNorm, ReLU, LeakyReLU, tanh, sigmoid, reshape
and a final softmax, run in float32 or float64 on the CPU. Transformers (GELU, LayerNorm, attention, residual
connections) are not supported yet; see [Limitations](#limitations) and, for where support would be added,
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md#where-to-change-the-code).

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [The input domain X](#the-input-domain-x)
4. [The norm](#the-norm)
5. [Method](#method)
6. [Why the bound is sound](#why-the-bound-is-sound)
7. [Scope and assumptions](#scope-and-assumptions)
8. [Results](#results)
9. [Tests](#tests)
10. [Limitations](#limitations)
11. [Project layout](#project-layout)
12. [References](#references)

## Install

```bash
pip install -e .                 # core: torch, numpy
pip install -e ".[hf,dev]"       # plus the Hugging Face example, the experiment runner and pytest
```

Requires Python 3.12 or newer (the pinned numpy needs it). Developed and tested with Python 3.12.10 and
CPU-only PyTorch 2.14.1 on Windows 10. The exact versions are pinned in `pyproject.toml`.

## Quick start

```bash
python examples/quickstart.py
```

This loads [`dacorvo/mnist-mlp`](https://huggingface.co/dacorvo/mnist-mlp) (Linear-ReLU-Linear-ReLU-Linear-softmax,
a model published as a reference for testing quantization) and an MNIST test image from
[`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist), both pinned to fixed revisions. It
quantizes the weights to 8 bits and bounds the change in the 10 output probabilities over all images within
0.003 (in pixel units) of the test image. Then it checks the bound by sampling and by a gradient attack.

The model file is fetched from the Hub and run with `trust_remote_code=True`. It is a short file that only
defines the three layers; read it before running.

The full experiments are driven by JSON configs:

```bash
python -m maxbound examples/configs/mnist_quant8.json --out results/mnist_quant8
python -m maxbound examples/configs/mnist_torch_compile.json --out results/mnist_torch_compile
```

`torch.compile` needs a C++ compiler. On Windows, run inside a Visual Studio developer environment
(`vcvars64.bat`) and set `TORCHINDUCTOR_CACHE_DIR` to a path without spaces.

## The input domain X

X describes the inputs of **one** example, without a batch dimension, **in the model's input units**.

| form | meaning |
|---|---|
| `Box(lower, upper)` | every x with `lower <= x <= upper` in each coordinate |
| `LinfBall(center, eps, clip=(lo, hi))` | every x with `abs(x - center) <= eps` in each coordinate, intersected with `[lo, hi]` |
| `{"type": "box", "lower": [...], "upper": [...]}` | the same as a dict (for JSON configs) |
| `{"type": "linf_ball", "center": [...], "eps": 0.1, "clip": [lo, hi]}` | |
| `(lower, upper)` | shorthand for a Box |

Every form is turned into a box, rounded outward so it always contains the region described. For the
MNIST model, which expects normalized inputs `(pixel - 0.1307) / 0.3081`, a ball of radius `eps` in pixel
units is `LinfBall((pixels - 0.1307) / 0.3081, eps / 0.3081, clip=(normalized 0, normalized 1))`.

## The norm

L-infinity over output coordinates: the bound covers `max_i |model(x)_i - cl_model(x)_i|` for every x in X.
If the model ends with softmax, the outputs are the probabilities; `MaxBound(..., output="logits")` bounds
the values before the softmax instead. `bound.per_output` gives the bound for each coordinate.

## Method

Full math and the soundness proof are in [docs/DESIGN.md](docs/DESIGN.md); a plain tour of the code is in
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

**1. Follow the difference, not the two models** (differential analysis, as in ReluDiff and NeuroDiff).
Bounding each model separately and subtracting is sound but very loose: if each output is known only to lie
in a range of width 4, their difference is known only to lie in a range of width 8, even for identical models.
We follow the original values `h` and the difference `d = h' - h` layer by layer. At a linear layer,

```
delta = z' - z = W' d + (W' - W) h + (b' - b)
```

At a ReLU, `relu(z + delta) - relu(z)` is exact when the neuron is surely on or surely off in each model
(four cases: `delta`, `0`, `-z`, `z'`). Otherwise it is enclosed by its values at the four corners of the
`(z, delta)` box; this is valid because the function is monotone in `delta` and, as ReLU is convex,
monotone in `z`. tanh and sigmoid use the mean value theorem.

**2. Affine forms that both networks share** (the zonotope domain of DeepZ). Each neuron is stored as
`c + sum_k G_k eps_k` plus a small extra radius, where the `eps_k` range over `[-1, 1]`. The first symbols are
the input coordinates. Because the two models see the same input, their forms share those symbols, and
subtracting cancels the shared part exactly; plain intervals cannot do that. Neurons that can switch inside
X get a new symbol. Consequences we test:
* identical models give a real-arithmetic bound of exactly 0, when their weights are used as stored. With
  BatchNorm, folding leaves a float64-sized remainder (about 1e-15), because each model's folded weights carry
  their own rounding radius;
* on a single input point, without a final softmax, the bound is the actual difference (up to float64 slack);
* if no ReLU can switch inside X, the bound on the outputs equals the true maximum.

**3. Floating point, twice.**
* *Our own float64 arithmetic* is made safe by rounding every bound outward. Matrix products get Higham's
  error bound `gamma_n (|W| |x| + |b|)`, which holds for any summation order.
* *The models' own float32 execution* differs from their exact real-number function. For each model,
  `fp_execution.py` bounds that drift over X layer by layer (Higham's bound for every dot product, Lipschitz
  constants for activations, and a derivation for softmax). The final bound is

```
real-arithmetic bound  +  allowance(original)  +  allowance(compiled)
```

  This is not optional: in 47 of 400 cases on the MNIST model, the observed float32 difference was larger
  than the real-arithmetic bound alone (see Results). Verifiers that ignore floating point can be fooled
  (Zombori et al. 2021; Jia and Rinard 2021).

**4. Softmax.** The softmax Jacobian has row sums `2 p_i (1 - p_i)`, and adding a constant to every logit
does not change softmax. So `|p_i' - p_i| <= max p_i (1 - p_i) * (max_j d_j - min_j d_j)`, with the probability
range taken along the segment between the two models' logits. For a confident classifier this is far
smaller than the global constant 1/2.

**5. Four analyses, smallest bound wins.** By default MaxBound runs zonotope-differential (the method
above), interval-differential, zonotope-separate and interval-separate (naive interval bound propagation),
and keeps the smallest bound per output. Each is sound, so the smallest is too, and the four values form an
ablation that `bound.method_bounds` reports.

**6. Certificate.** `bound.same_prediction` is a class `c` if, for every input in X, both executed models
provably rank `c` strictly first in the bounded outputs, else `None`. The bounded outputs are the
probabilities when the model ends with softmax. They are compared as probabilities, not as logits, because
rounding can turn two different logits into equal probabilities and change the argmax.

## Why the bound is sound

1. X is inside the analysed box, by outward rounding.
2. **Invariant:** for every input there is one assignment of the shared noise symbols (and a small extra
   error per neuron) that makes every form equal the true value. This holds at the input and is preserved by
   every layer rule:
   * affine layers are exact on forms, and their float64 rounding is added to the extra radius;
   * activations are exact, or enclosed with a new symbol that is used nowhere else.
3. So at the output, every true difference lies in the range of its form. With a final softmax, the lemma in
   section 4 converts the forms into a bound on probabilities.
4. Under the assumptions below, each model's float32 output is within its allowance of its real output.
5. The triangle inequality adds the three parts. The minimum over analyses is sound because each analysis is.

Sampling and the attack are not part of this argument. They can refute a bound, never prove one, and serve
as the sanity check.

## Scope and assumptions

**Supported models:** a single chain of Linear, Conv2d, eval-mode BatchNorm (folded, so fusion passes are
verified rather than assumed), ReLU, LeakyReLU (slope in [0, 1]), tanh, sigmoid, flatten/reshape, dropout/identity,
and a final softmax. Weights in float32 or float64, execution on the CPU. Residual connections, LayerNorm,
attention and pooling raise `UnsupportedModelError`.

**Supported compile passes:** anything that keeps the layer structure and changes weights (weight
quantization, pruning, casting, BatchNorm fusion), and `torch.compile`. If the structure changes, the
differential analyses are skipped and the separate ones still give a sound, looser bound.

**Assumptions** (also in `bound.assumptions`):

* **A1:** IEEE round-to-nearest, with matrix products and convolutions at full precision (no TF32 or bfloat16),
  in any order, with or without fused multiply-add. Every operation keeps the model's dtype.
* **A2:** float32 `tanh`, `sigmoid` and `exp` are accurate to 16 units of roundoff; float64 ones to `2^-45`
  relative. Both are tested against 50-digit decimal arithmetic on grids of inputs. The allowance uses the
  constant that matches each model's format.
* **A3:** eval-mode BatchNorm executes as `x * alpha + (beta - mean * alpha)` with
  `alpha = weight / sqrt(var + eps)`, each step rounded once, and `eps` used exactly or rounded to the
  model's format. The allowance covers the rounded denominator, underflow in `alpha` and its amplification
  by `x`. Denominators that could round to zero are refused (DESIGN.md section 8).
* **A4:** softmax executes as `exp(x - max) / sum`. This is what the code Inductor generates for the MNIST
  model does.
* **A5:** no overflow.
* **A6:** each model computes exactly the chain `torch.fx` traces. A `torch.compile` model computes the same
  real-number function as its original module: Inductor fuses and reorders, and on Windows compiles without
  fast-math flags (checked in `torch/_inductor/cpp_builder.py`). We did not verify every generated kernel.

**What is enforced by code** (the rest are assumptions):
* `check_runtime` refuses:
  * float32 matmul precision other than "highest";
  * any `fp32_precision` switch (global, and oneDNN matmul, conv and rnn) other than IEEE;
  * active CPU autocast;
  * parameters or buffers outside the CPU;
  * Inductor's unsafe-math and fast-math switches, read from the live config and from the environment.
* Extraction refuses:
  * any operation whose output dtype differs from the model's;
  * any unsupported operation;
  * BatchNorm with a negative running variance.
* The analysis stops with an error on possible overflow, or on values outside the range where its rounding
  rules hold.

## Results

All numbers below come from the files in [results/](results/) and the tests named. Pixel values are in [0, 1].
"Real part" is the real-arithmetic bound, "largest found" the best of 1000 random samples and a 50-step
gradient attack per image. Values are medians over 50 MNIST test images; timings are per `MaxBound` call
on the development CPU.

**Hand-checkable cases** ([tests/test_hand_cases.py](tests/test_hand_cases.py), real arithmetic):

| case | true max | zonotope diff. | interval diff. | zonotope sep. | interval sep. |
|---|---|---|---|---|---|
| `x1 + x2` vs `1.1 x1 + 0.9 x2` on `[0, 1]^2` | 0.1 | 0.1 | 0.1 | 0.1 | 2.0 |
| `relu(x)` vs `relu(1.5 x)` on `[-1, 1]` | 0.5 | 0.5 | 0.5 | 1.0 | 1.5 |
| two ReLU neurons, one switching from on to off between the models, on `[1, 2]` | 4 | 4 | 4 | 4 | 5 |
| the same with one neuron switching from off to on | 4 | 4 | 5 | 4 | 5 |

**8-bit per-channel weight quantization, logits** ([results/mnist_quant8.md](results/mnist_quant8.md)):

| eps | bound | real part | fp allowance | largest found | real / found | bound / found | zonotope diff. | interval diff. | zonotope sep. | interval sep. |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.229 | 0.0278 | 0.218 | 0.0278 | 1.00 | 8.03 | 0.0278 | 0.0278 | 0.0278 | 0.0278 |
| 0.001 | 0.232 | 0.0295 | 0.219 | 0.0282 | 1.03 | 7.97 | 0.0295 | 0.0773 | 0.0347 | 4.58 |
| 0.003 | 0.239 | 0.0354 | 0.219 | 0.0290 | 1.17 | 7.92 | 0.0354 | 0.187 | 0.0924 | 13.7 |
| 0.01 | 0.280 | 0.0700 | 0.221 | 0.0324 | 2.12 | 8.58 | 0.0700 | 0.642 | 0.879 | 44.2 |

The two ratio columns are medians over images of the per-image ratio. "real / found" shows how tight the
analysis is; "bound / found" shows the bound you actually get, which includes the worst-case rounding
allowance (per image it ranges from 4.2x to 17.4x).

**The same, probabilities (what the model returns):**

| eps | bound | real part | fp allowance | largest found | real / found | bound / found | zonotope diff. | interval diff. | zonotope sep. | interval sep. |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.05e-5 | 4.32e-7 | 2.00e-5 | 1.19e-7 | 3.04 | 150 | 4.32e-7 | 4.32e-7 | 4.32e-7 | 4.32e-7 |
| 0.001 | 2.11e-5 | 5.57e-7 | 2.05e-5 | 2.56e-7 | 2.40 | 92.1 | 5.57e-7 | 1.55e-4 | 7.02e-7 | 8.36e-3 |
| 0.003 | 2.36e-5 | 9.81e-7 | 2.23e-5 | 3.58e-7 | 2.55 | 85.4 | 9.81e-7 | 0.0849 | 4.15e-6 | 6.42 |
| 0.01 | 6.69e-5 | 1.19e-5 | 5.49e-5 | 5.96e-7 | 10.3 | 114 | 1.19e-5 | 0.31 | 1.67e-4 | 21.4 |

The model is confident on these images, so the softmax bound (which uses `p(1 - p)`) makes the probability
bounds 4,000 to 11,000 times smaller than the logit bounds (for example 2.05e-5 against 0.229 at eps = 0). Values above 1 in the last column show how loose naive
intervals are; a difference of probabilities is never more than 1. Much of what sampling finds on the
probabilities is float32 rounding (1.19e-7 is one unit in the last place at 1.0).

* No violations in 400 checks (50 images, 4 radii, logits and probabilities).
* Both models provably predict the same class for every input within the radius for 49 of 50 images. The
  exception is test image 33, a "4", for eps >= 0.001.
* One call takes about 0.2 s (four analyses and two rounding passes).

**Rounding matters.** In 47 of the 400 cases, the float32 difference that was actually observed exceeded
the real-arithmetic part of the bound (by up to 19.9x for probabilities). An analysis in real arithmetic
alone would have returned a wrong guarantee in those cases. With the allowance there are no violations.

**`torch.compile`** ([results/mnist_torch_compile.md](results/mnist_torch_compile.md)):

* The compiled model is not bit-for-bit identical to the original. Its probabilities differed in 43 of 100
  checks (at most 3.0e-8). Its logits never differed in our checks: the generated code calls the same
  `addmm` matrix kernels, so the difference arises in Inductor's fused softmax kernel.
* **This result rests on assumption A6.** We analyse the compiled model through its original graph, assuming
  Inductor computes the same real-number function and only rounds differently. We support that with
  Inductor's compile flags and the generated softmax code, not by verifying every generated kernel. Under A6,
  the real-arithmetic part is 0 by construction, and the whole bound is the two rounding allowances: a median
  of 2.0e-5 (eps = 0) and 5.5e-5 (eps = 0.01) on the probabilities, and 0.22 on the logits.
* The bound held in all 200 checks, but it is very loose here. Even the smallest bound is 438 times the
  largest deviation observed in any check. Per image the ratio is in the millions or more, because most observed
  deviations are at the level of float32 noise. This is the price of the worst-case rounding analysis.
* The separate analyses do not reach 0 here (0.84 on the logits at eps = 0.01 with zonotopes, 44.2 with
  intervals), which shows again why following the difference matters.

**Planted-bug check** ([results/mutation_check.md](results/mutation_check.md)): `tools/mutation_check.py`
plants 18 known soundness bugs one at a time, for example forgetting the `(W' - W) h` term, rounding inward,
using the wrong corners, or halving the softmax bound. The test suite catches all 18. The first version had
14 bugs and its first run caught 12. The two that slipped through (missing float64 slack in linear layers,
missing softmax rounding) led to two new tests.

**Independent review.** A separate review of an earlier version found three inputs where the bound was too
small, all fixed:
* BatchNorm whose precomputed scale is subnormal in float32 (the scale's underflow error is then multiplied
  by a large input);
* a LeakyReLU slope that is not exactly representable in float32;
* a domain given as a Python list, which was converted to float32 before float64.

It also found:
* a prediction certificate that compared logits where rounded probabilities could tie;
* a tanh slope bound that overflowed far from 0;
* unchecked convolution precision and dtype casts.

Each is fixed, kept as a regression test (`tests/test_review_regressions.py`) and, for the first four, as a
planted bug.

## Tests

```bash
pytest                        # all tests (10 to 20 s; the network test downloads about 4 MB once)
pytest -m "not network"       # skip the test that downloads from the Hugging Face Hub
python tools/mutation_check.py
```

The test of `torch.compile` runs when a C++ compiler is on PATH and is skipped otherwise. The suite
checks:
* the hand cases above;
* exactness where the true maximum is known;
* the empirical check (sampling and attack never beat any analysis), over 4 activations, 2 compile changes,
  with and without softmax, float64 models, and a CNN with BatchNorm;
* every activation rule on fine grids;
* rounding against exact rational arithmetic;
* the float32 allowance against measured float32 versus float64 gaps;
* model extraction and BatchNorm fusion;
* every counterexample from the independent review.

[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) maps each test file to what it checks.

## Limitations

* **The float32 allowance is pessimistic.** It assumes every rounding error goes the worst way. On the MNIST
  logits it is about 0.22, against a real-arithmetic part of 0.03 to 0.07. For `torch.compile` it is several
  hundred times the deviation observed. The real-arithmetic part is reported separately for this reason.
* **Precision drops as more neurons can switch.** At eps = 0.01 the logit bound is about 2.1x the largest
  difference found; for larger regions it grows further.
* **Model scope:** single-chain networks only; no residuals, attention, LayerNorm, pooling or GELU yet.
* **No activation quantization:** int8 inference kernels that quantize activations at runtime are not modelled.
* **CPU float32 or float64 only**, and `torch.compile` under assumption A6.
* **Memory:** the generator matrix has one row per input coordinate and per new symbol, which limits large
  convolutional networks.
* **One region per call:** batch over images by calling `MaxBound` per image (as `experiment.py` does).

## Project layout

```
src/maxbound/
  api.py           MaxBound, Bound: the entry point
  config.py        MaxBoundConfig (dict / JSON settings)
  domain.py        Box, LinfBall: the input region X
  graph.py         torch.fx tracing into a chain of layers, BatchNorm folding, alignment
  zonotope.py      shared affine forms and their arithmetic
  activations.py   ReLU / LeakyReLU / tanh / sigmoid rules
  analysis.py      the layer loop: four analyses
  softmax.py       bounds through a final softmax
  fp_execution.py  the models' float32 rounding allowance
  rounding.py      outward rounding, Higham's gamma_n
  intervals.py     float64 interval arithmetic
  compilers.py     example compile passes (quantize, cast, prune, fuse BatchNorm, torch.compile)
  empirical.py     sampling and gradient attack (sanity checks)
  experiment.py    JSON-config runner (python -m maxbound)
examples/          quickstart.py, configs/*.json
results/           tables and logs of the runs above, planted-bug report
tests/             pytest suite
tools/             mutation_check.py
docs/              DESIGN.md (math, proof), WALKTHROUGH.md (code tour, review Q&A)
```

## References

* N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002 (Section 3.1).
* G. Singh, T. Gehr, M. Mirman, M. Puschel, M. Vechev. Fast and Effective Robustness Certification. NeurIPS 2018.
* B. Paulsen, J. Wang, C. Wang. ReluDiff: Differential Verification of Deep Neural Networks. ICSE 2020.
* B. Paulsen, J. Wang, J. Wang, C. Wang. NeuroDiff: Scalable Differential Verification of Neural Networks
  using Fine-Grained Approximation. ASE 2020.
* D. Zombori, B. Banhelyi, T. Csendes, I. Megyeri, M. Jelasity. Fooling a Complete Neural Network Verifier. ICLR 2021.
* K. Jia, M. Rinard. Exploiting Verified Neural Networks via Floating Point Numerical Error. SAS 2021.
