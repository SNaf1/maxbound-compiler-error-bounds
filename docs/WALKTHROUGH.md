# maxbound walkthrough

A plain-language tour of the code: what each part does and why, what happens in one call, where to change
things, what can go wrong, and how each part is tested. The math and the soundness argument are in
[DESIGN.md](DESIGN.md); this file is the map.

## The idea in five sentences

1. We want a number that is guaranteed to be at least as large as the biggest output difference between a
   model and its compiled version, for every input in a region X.
2. Instead of bounding each model separately and subtracting, which is sound but very loose because the two
   uncertainties add up, we follow the *difference* between the two models through the network, layer by layer.
3. Each neuron is stored as a formula in the input ("constant plus weighted sum of how far each input
   moved"), and both models use the same input terms, so the shared part cancels exactly when we subtract.
4. Our own calculations are done in float64 with every result rounded outward, and the models' float32
   execution is covered by a separate, worst-case rounding bound.
5. Sampling and a gradient attack check the result: they may never find a bigger difference than the bound.

## Module map

| file | job | main functions |
|---|---|---|
| `api.py` | the public entry point | `MaxBound`, `Bound`, `check_runtime` |
| `config.py` | settings from a dict or JSON | `MaxBoundConfig` |
| `domain.py` | the input region X | `Box`, `LinfBall`, `as_box`, `domain_from_config` |
| `graph.py` | model to a list of layers | `extract`, `fold_batchnorm`, `alignment_problem`, `unwrap_compiled`, `AffineOp` |
| `zonotope.py` | the formulas ("affine forms") and their arithmetic | `Zonotope`, `affine`, `add`, `sub`, `scale`, `shift`, `select`, `fresh` |
| `activations.py` | the math of ReLU, LeakyReLU, tanh, sigmoid | `value_range`, `deriv_range`, `relaxation`, `diff_range` |
| `analysis.py` | the loop over layers (four variants) | `run`, `_diff_step`, `_act_value`, `_act_diff` |
| `softmax.py` | bounds through a final softmax | `pair_bounds`, `prob_range`, `difference_bound` |
| `fp_execution.py` | the models' own float32 rounding | `execution_allowance` |
| `rounding.py`, `intervals.py` | keeping our float64 arithmetic safe | `up`, `down`, `inflate`, `gamma`, `Interval` |
| `compilers.py` | example compile passes | `quantize_weights`, `cast_weights`, `prune_magnitude`, `fuse_batchnorm`, `torch_compile` |
| `empirical.py` | sampling and gradient attack (checks only) | `sampled_max_difference`, `attack_max_difference` |
| `experiment.py` | JSON-config runner, results tables | `run`, `summarize`, `to_markdown`, `main` |
| `errors.py` | the two exception types | `UnsupportedModelError`, `NumericalInconsistency` |

## One call, step by step: `MaxBound(model, cl_model, X)`

1. **Settings.** `MaxBoundConfig.build` merges defaults, a config dict or JSON file, and keyword overrides.
2. **Region.** `as_box(X)` turns a `Box`, `LinfBall`, dict or `(lower, upper)` pair into a `Box`, rounding outward
   so the box contains every point of X.
3. **Safety checks.** `check_runtime` refuses PyTorch settings that would break the rounding assumptions:
   * reduced-precision float32 matmuls or convolutions (TF32 or bfloat16, any `fp32_precision` switch);
   * active CPU autocast;
   * parameters or buffers outside the CPU;
   * unsafe fast-math in `torch.compile` (live config and environment).
4. **Models to layer lists.** `extract` traces each model with `torch.fx`, maps every operation to an
   `AffineOp`, `ActOp`, `ReshapeOp` or `SoftmaxOp`, copies weights to float64, and records shapes. For a
   `torch.compile` model it analyses the original module (`unwrap_compiled`).
5. **Line them up.** `fold_batchnorm` merges BatchNorm into the layer before it; `alignment_problem` checks the
   two lists have the same structure. If not, only the "separate" analyses run.
6. **Real-arithmetic bound.** For each of the four analyses, `analysis.run` walks the layers:
   * affine layer: `z = W h + b` for the original and `delta = W' d + (W' - W) h + (b' - b)` for the difference;
   * activation: `_act_value` for the original's new values, `_act_diff` for the difference (exact when the
     neuron is surely on or off in both models, an enclosing range otherwise);
   * reshape: reshape the formulas.
   Then `_real_bound` turns the final formulas into one number per output (through `softmax.difference_bound`
   if the model ends with softmax). The smallest value per output across the analyses is kept.
7. **Rounding allowance.** `execution_allowance` bounds, for each model separately, how far float32
   execution can drift from the exact function, layer by layer.
8. **Assemble.** bound = real part + allowance(original) + allowance(compiled), rounded up. `_certify`
   checks whether both models must rank the same class first everywhere in X, comparing probabilities
   (not logits) when the model ends with softmax. Everything is packed into a `Bound`.

## File by file

### `domain.py`
* `Box(lower, upper)` stores float64 bounds and checks shape, finiteness and `lower <= upper`.
  `float32_bounds` finds the smallest and largest float32 values inside the box (the inputs that can
  actually reach a float32 model); `sample` draws float32 points inside, a quarter of them at random corners
  because differences are often largest there.
* `LinfBall(center, eps, clip)` is "center plus or minus eps in every coordinate", optionally clipped to the
  valid input range; `to_box` rounds `center - eps` down and `center + eps` up.
* X has **no batch dimension** and is given **in the model's input units** (for MNIST, normalised pixels).

### `graph.py`
* `extract` is the bridge between PyTorch and the analysis. It traces with `torch.fx`, runs the traced graph
  on two copies of an example input to learn every shape (two, so a reshape that would mix examples is
  caught), and walks the nodes. It insists on a single chain: every operation must take exactly the previous
  result (shape queries like `x.size(0)` are ignored), and every operation must keep the model's dtype.
  `_convert` maps one node to an op.
* `AffineOp` covers Linear, Conv2d and BatchNorm ("scale"). It knows how to apply `W`, `|W|` and the weight
  radius to a batch (`linear`, `abs_linear`, `err_linear`) and its `fan_in` (how many products feed one
  output, used in the rounding constants). `difference` builds the op with `W' - W` and `b' - b`.
* `_bn_op` turns eval-mode BatchNorm into `s * x + t` and records the float64 rounding of `s` and `t`.
* `fold_batchnorm` / `_fold` merge BatchNorm into the previous Linear/Conv, again recording rounding.
* `unwrap_compiled` returns the original module inside a `torch.compile` result.

### `zonotope.py`
* A `Zonotope` holds `c` (constant per neuron), `G` (one row per noise symbol), `e` (extra radius per neuron).
  Value of a neuron = `c + sum_k G[k] * eps_k + xi`, with every `eps_k` in [-1, 1] and `|xi| <= e`.
* `fresh` creates new symbols ("this neuron is somewhere in `[m - r, m + r]`"); without a symbol pool it puts
  the radius into `e`, which turns everything into plain interval arithmetic (the "interval" mode).
* `affine` pushes forms through a layer: `c` and every row of `G` go through `W`, `e` goes through `|W|`, and
  the rounding of our own float64 matrix products is added to `e` (Higham's bound).
* `add`, `sub`, `scale`, `shift` are the obvious operations plus a tiny rounding slack; `select` picks, per
  neuron, one of two forms (used for the case analysis of ReLU).
* `bounds` gives the range of a form: `c` plus or minus `(sum of |G| + e)`.

### `activations.py`
* `sigma_bounds` gives safe lower and upper values of an activation at a point.
* `value_range`: all four activations are nondecreasing, so the range over `[l, u]` is `[sigma(l), sigma(u)]`.
* `deriv_range`: slopes of tanh/sigmoid are largest nearest 0, smallest at an endpoint. They are computed
  from `exp(-|x|)` so that nothing overflows far from 0.
* `relaxation`: a line plus a band that contains the activation over an interval (DeepZ's rule for ReLU).
* `diff_range`: the range of `sigma(z + delta) - sigma(z)`. ReLU: check the four corners (justified by
  monotonicity and convexity). tanh/sigmoid: mean value theorem. Both are intersected with the plain
  "range minus range" enclosure, and an exactly zero `delta` gives exactly zero.

### `analysis.py`
* `METHODS` lists the four analyses: zonotope or interval, differential or separate.
* `run` sets up the input formulas and loops over the layers; `_diff_step` does one layer for the pair
  (original values `h`, difference `d`); `_run_single` does one network alone.
* `_act_value`: ReLU neurons that are surely on are copied, surely off become zero (or `a z` for LeakyReLU),
  the rest get the DeepZ line plus a new symbol. tanh/sigmoid always get a line plus a new symbol.
* `_act_diff`: the four exact ReLU cases from the table in DESIGN.md section 5.2, otherwise the enclosing
  range from `diff_range` as a new symbol. For tanh/sigmoid it keeps `kappa * delta` plus a band, or the
  enclosing range, whichever is narrower.

### `softmax.py`
* `pair_bounds` bounds `v_i - v_j` for every pair of outputs directly from the formulas (shared terms cancel).
* `prob_range` turns pairwise logit differences into a range for each probability.
* `difference_bound` applies the softmax lemma: `|p_i' - p_i| <= max p(1-p) * (largest d_j - d_k)`.

### `fp_execution.py`
* `execution_allowance` walks one model's layers and tracks `a`, a bound on |float32 value - exact value|:
  each Linear/Conv adds Higham's dot-product rounding bound and carries the old error through `|W|`, ReLU
  carries it unchanged, LeakyReLU adds the rounding of its slope and of the multiplication, tanh/sigmoid
  carry it with their Lipschitz constant and add their own rounding. BatchNorm is `_batchnorm_terms`, which
  includes underflow of its precomputed scale. Softmax is `_softmax_terms` (derivation in DESIGN.md
  section 8).
* Value sizes come from a zonotope analysis of that one model, which keeps the allowance as small as the
  worst-case model allows.

### `rounding.py` and `intervals.py`
* `up`/`down` move a float64 value one representable number up/down (zeros stay zero). `gamma(n, u)` is the
  standard rounding constant for sums of `n` products. `inflate`/`deflate` turn computed nonnegative sums
  into guaranteed upper/lower bounds. `check_range` stops the analysis on values where these rules would fail.
* `Interval` is a pair of float64 tensors with outward-rounded `add`, `sub`, `scale`, `mul`, `hull`,
  `intersect` (which raises if two enclosures that must overlap do not).

### `api.py`, `config.py`, `errors.py`
* `MaxBound` is the step list above. `Bound` is a `float` subclass, so `bound` works as a number, and
  carries `real_bound`, `fp_allowance`, `per_output`, `method_bounds`, `same_prediction`, `assumptions` and
  `summary()`.
* `MaxBoundConfig` fields: `method` ("best" or one analysis), `output` ("model" or "logits"), `fp_model`
  ("auto", "float32", "float64" or "real"), `check_runtime`.
* Logging: `api.py` reports to the `maxbound.api` logger:
  * what was traced from each model;
  * a warning when the models do not line up and only the separate analyses run;
  * each analysis's bound;
  * the two rounding allowances;
  * the final bound.

  Warnings show by default; `logging.basicConfig(level=logging.INFO)` shows the rest. The experiment CLI
  logs one progress line per image, or every step with `--verbose`.

### `compilers.py`, `empirical.py`, `experiment.py`
* Compile passes return a changed copy and never touch the original. `quantize_weights` uses a symmetric
  per-channel grid; `fuse_batchnorm` uses PyTorch's own fusion helpers; `torch_compile` wraps `torch.compile`.
* `sampled_max_difference` and `attack_max_difference` search for big differences; they can only produce
  lower bounds. `without_final_softmax` gives a logits version of a model for checking logit bounds.
* `experiment.run` executes a JSON config (model, images, compiler, radii, settings) and records, for every
  image and radius, the bound, its parts, what sampling and the attack found, and the certificate.

## Where to change the code

| to support... | change |
|---|---|
| a new linear layer (e.g. average pooling) | `graph._convert`: build an `AffineOp` with a new `kind`; `AffineOp._apply` and `fan_in`. The affine rules in `zonotope.affine` and `fp_execution` then work unchanged. |
| a new activation (e.g. GELU) | `activations.py` (value range, slope range, relaxation, difference range; GELU is not monotone, so its range needs the minimum at x near -0.75), `fp_execution._LIPSCHITZ` and its own-rounding term, and the mapping in `graph._convert`. |
| a model with branches (residual add) | `graph.extract` would keep a value per graph node instead of a chain; an add node is `zonotope.add` of the two forms. The layer rules stay the same. |
| a new Hugging Face model | nothing if it is a single chain of supported layers; set `model.id`/`revision` in the config. |
| a new dataset | `experiment.load_images` (expects a parquet file with `image` and `label` columns) and the `inputs` section of the config. |
| a new compile pass | add a function to `compilers.py` and register it in `experiment.COMPILERS`. If it changes the layer structure, MaxBound falls back to the separate analyses automatically. |
| another norm | only the last step: from the per-output bounds, L-infinity is their max; an L2 bound would be the square root of the sum of squares (sound, looser). |
| GPU execution | the rounding model in `fp_execution.py` assumes CPU float32 semantics; GPU kernels may use TF32 and would need their own assumptions. |

## Failure modes and edge cases

| situation | what happens |
|---|---|
| unsupported operation (LayerNorm, attention, pooling, skip connection) | `UnsupportedModelError` naming the operation |
| `torch.fx` cannot trace (data-dependent control flow) | `UnsupportedModelError` with the tracer's message |
| model in training mode | `ValueError` asking for `model.eval()` |
| the two models have different structure | differential analyses skipped, separate ones used, reason in `bound.alignment_problem` |
| X does not match the model's input shape, or the clip range misses the ball | `ValueError` |
| a float32 model and an X with no float32 value in some coordinate | the bound is still computed (the analysis does not need float32 points); only `Box.sample`, used by the empirical checks, raises `ValueError` |
| BatchNorm with negative running variance, or `var + eps` that could round to zero | `UnsupportedModelError` / `NumericalInconsistency` |
| a value becomes infinite, NaN, or so small or large that rounding rules fail | `NumericalInconsistency` (we stop rather than return an unproven number) |
| two enclosures that must overlap do not | `NumericalInconsistency` (signals a bug) |
| possible float32 overflow inside a model | `NumericalInconsistency` |
| TF32/bfloat16 matmuls or convolutions, CPU autocast, GPU parameters or buffers, unsafe fast-math in torch.compile | `RuntimeError` from `check_runtime` |
| float16 / mixed-precision models, or any operation that changes dtype (e.g. `softmax(..., dtype=float16)`) | `UnsupportedModelError` (no rounding model for them) |
| identical models | real part exactly 0 (about 1e-15 with BatchNorm folding), bound = the two rounding allowances |
| X is a single point (eps = 0) | without softmax, the real part equals the actual real difference; with softmax it is an upper bound only |
| X given as a Python list or JSON | converted straight to float64, so values that are not float32 numbers are kept exactly |
| float64 models | supported; X does not need to contain a float32 value |
| softmax not at the end | `UnsupportedModelError` |
| very large models | no special handling: memory grows with (input size + new symbols) times layer size |

## How each part is tested

| test file | what it checks |
|---|---|
| `test_rounding_intervals.py` | outward rounding, `inflate`/`deflate`, interval operations and a full affine layer against exact rational arithmetic (`fractions.Fraction`) |
| `test_activations.py` | every activation rule contains the true values on fine grids; zero difference stays zero |
| `test_hand_cases.py` | pen-and-paper cases: linear (0.1 vs 2.0), ReLU (0.5 / 0.5 / 1.0 / 1.5), the two switching ReLU cases (exactly 4), identical models (exactly 0) |
| `test_exactness.py` | equality with the true maximum when no ReLU can switch; a single point reproduces the forward pass |
| `test_soundness.py` | sampling and attack never beat any analysis, for 4 activations x 2 compile changes x softmax or not, float64 models, a CNN with BatchNorm; monotonicity of the interval method |
| `test_fp_execution.py` | the float32 allowance covers measured float32-vs-float64 gaps (MLPs, softmax alone, BatchNorm, CNN); float32 and float64 library functions meet the stated accuracy (50-digit decimal reference) |
| `test_softmax_and_certificate.py` | tight softmax case (`eta / 2`), the confidence effect, certificate given / refused |
| `test_graph.py` | extraction of sequential and functional models, rejection of unsupported ones, fallback for misaligned models, BatchNorm fusion verified |
| `test_domain.py`, `test_compilers.py` | domains and compile passes |
| `test_real_models.py` | the Hugging Face model (marked `network`), `torch.compile` (needs a C++ compiler), the experiment summary |
| `test_review_regressions.py` | the counterexamples found in a second pass over the work, which broke an earlier version: subnormal BatchNorm scale, LeakyReLU slope rounding, list domains, a softmax tie in the certificate, tanh/sigmoid slopes far in the tail, reduced-precision convolution, dtype-changing softmax, float64 points |
| `tools/mutation_check.py` | plants 18 known soundness bugs one at a time and checks that the suite catches each (`results/mutation_check.md`) |

## Questions to expect, with short answers

**Why not just sample?** Sampling finds inputs where the models differ; it cannot show that no input differs
more. The task asks for a guarantee, so sampling is only our sanity check.

**What exactly does "sound" mean here?** For every input in X (after outward rounding), the actual float32
output difference is at most the returned number, provided the stated assumptions hold (DESIGN.md section 9).

**Why follow the difference instead of each model?** If each model's output is only known to lie in a range
of width 4, their difference is only known to lie in a range of width 8, even if the models are identical.
Following the difference gives exactly 0 for identical models whose weights are used as stored (about 1e-15
when BatchNorm is folded) and stays small when the change is small.

**Why zonotopes rather than intervals?** Intervals forget which input caused which value, so errors add up at
every layer. With shared symbols the two models' dependence on the input cancels. On the MNIST model at
eps = 0.01 the difference bound on the logits is 0.070 with zonotopes, 0.642 with intervals, and 44.2 when
each model is bounded separately with intervals.

**Why not CROWN or an off-the-shelf verifier?** We did not run CROWN, so we make no claim about its numbers.
A single-network verifier applied to both models side by side relaxes each model's ReLUs on their own and
never uses the fact that the two models are close. Our closest measured analogue is the "zonotope-separate"
row (separate relaxations, shared input): 0.879 at eps = 0.01 against 0.070 for the differential version.
CROWN uses different (backward, linear) relaxations, so that row is not a CROWN result. ReluDiff and NeuroDiff
report the same lesson for their own baselines. CROWN-style linear bounds applied to the difference would be a
natural next step.

**Why is the float32 allowance so large?** It assumes every rounding error in every one of the hundreds of
thousands of multiply-adds goes the worst way and adds up. That is the price of a guarantee that does not
depend on luck. We report it separately, so the real-arithmetic part (which is tight) is visible.

**Do we even need it?** Yes. On the MNIST model, in 47 of 400 cases the observed float32 difference was larger
than the real-arithmetic bound alone. Ignoring rounding would have produced wrong guarantees.

**What is not verified?** That `torch.compile`'s kernels keep the real-number function (we checked the compile
flags and the generated softmax, not every kernel), the library functions' accuracy beyond the tested grids,
and BatchNorm's execution order. All are listed as assumptions.

**Why is the zonotope bound not monotone in eps?** The line used for a switching ReLU depends on the neuron's
range, so a larger region can occasionally give a slightly better line. Interval arithmetic is monotone, and
the tests check that.

**How long does it take?** About 0.2 seconds per call for the MNIST model, for all four analyses together.
