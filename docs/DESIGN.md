# maxbound: design and soundness argument

This document gives the math behind every step of `MaxBound(model, cl_model, X)` and the argument that the
returned number is a sound upper bound. The code mirrors the sections below; each section names the file.

Notation: `f` is the original model, `g` the compiled model. A tilde means the exact real-number function a
model describes (`f~`), a hat means what the model actually computes in floating point (`f^`). `u` is the
unit roundoff of a format: `u = 2^-24` for float32, `2^-53` for float64. `|A|` is the elementwise absolute
value of a matrix, and inequalities between vectors hold elementwise.

---------------------------------------------------------------------------------------------------

## 1. What is bounded

For an input region `X` the package returns a number `B` with

```
max over x in X of  || g^(x) - f^(x) ||_inf   <=   B
```

The norm is L-infinity over output coordinates: the largest absolute difference in any single output
value. If the models end with softmax, the outputs are probabilities (or logits, with `output="logits"`).

## 2. How the bound is put together  (api.py)

By the triangle inequality, for every x and every output i,

```
|g^_i - f^_i|  <=  |g~_i - f~_i|  +  |g^_i - g~_i|  +  |f^_i - f~_i|
                   real part R_i      allowance A_g,i   allowance A_f,i
```

* `R` bounds the difference of the two real-number functions over X (sections 4 to 6). It is computed by
  four analyses; each is sound, so the smallest value per output is sound too.
* `A_f` and `A_g` bound how far each model's floating-point execution drifts from its real-number function
  over X (section 8).
* `B = max_i (R_i + A_g,i + A_f,i)`, with the sum rounded up.

## 3. From a model to a chain of operations  (graph.py)

`torch.fx` records the operations of `forward()`. Supported models are a single chain
`x -> op_1 -> op_2 -> ... -> op_L`, where each op is

* affine: `y = W x + b` (Linear, Conv2d, or eval-mode BatchNorm as a per-channel scale and shift),
* an elementwise activation: ReLU, LeakyReLU with slope `a` in [0, 1], tanh, sigmoid,
* a reshape, or a final softmax over a vector.

Weights are converted from float32 to float64, which is exact. Eval-mode BatchNorm is `s * x + t` with
`s = w / sqrt(var + eps)` and `t = beta - mean * s`. Because `s (W x + b) + t = (s W) x + (s b + t)` holds
exactly in real arithmetic, a BatchNorm is folded into the affine op before it. The float64 rounding of
`s`, `t`, `s W` and `s b + t` is recorded as a radius around each weight and bias (`weight_err`,
`bias_err`) and handled in section 4. Folding makes "Linear + BatchNorm" in the original line up with the
single fused Linear that a compiler produces, so a fusion pass is verified rather than assumed.

Two chains *line up* if they have the same op types, shapes and activation settings; weights may differ.
Then the differential analyses apply; otherwise only the separate ones do.

A model returned by `torch.compile` is analysed through its original module (`_orig_mod`), under the
assumption that Inductor keeps the real-number function and changes only rounding (section 9, A6).

## 4. Affine forms  (zonotope.py)

Every neuron value is represented as an affine form

```
v  =  c + sum_k G_k eps_k + xi ,        eps_k in [-1, 1],   |xi| <= e
```

The first symbols stand for the input coordinates: if X is inside the box `[c0 - r0, c0 + r0]`, then
`x_j = c0_j + r0_j eps_j`. New symbols are added when a nonlinear step must be approximated.

**Invariant.** For every input `x` in X there is one assignment of all symbols `eps` in `[-1, 1]`, shared by
every form, and for each neuron some `|xi| <= e`, such that every form equals the true value of its neuron.

* *Affine op.* `W v + b = (W c + b) + sum_k (W G_k) eps_k + W xi`. The first two parts are exact in real
  arithmetic, and `|W xi| <= |W| e`. In float64 the computed `W c + b` and `W G_k` are off by at most
  `gamma_{n+1} (|W| |c| + |b|)` and `gamma_n |W| |G_k|` (Higham 2002, Section 3.1, for any summation order
  and with fused multiply-add). Summed over the symbols, the second is at most `gamma_n |W| sum_k |G_k|`.
  The new `e` is `|W| e + gamma_{n+1} (|W| |c| + |W| sum_k |G_k| + |b|)`, plus `|dW| |x| + |db|` when the
  weights carry a radius (`dW`, `db`), with `|x| <= |c| + sum_k |G_k| + e`. It is rounded up (section 7).
* *Sum, difference, scaling by a constant.* Exact on `c` and `G` up to one rounding per entry, which is at
  most `2u` times the computed entry and zero when one operand is zero. That slack is added to `e`.
* *Concretization.* The range of a form over all symbols is `c +- (sum_k |G_k| + e)`. This is exact for
  the form, because each symbol can move independently.
* *Fresh symbol.* "This neuron lies in `[m - r, m + r]`" is the form `m + r eps_new`. Since `eps_new`
  appears nowhere else, it can be chosen per input to match the true value, so the invariant is kept.

Without new symbols (every form collapsed to `c +- e` after each step) this is plain interval arithmetic.
That is the "interval" mode of the engine.

**Why shared symbols matter.** The original and the compiled network see the same input, so their forms
use the same input symbols. When we subtract forms the shared part cancels exactly; intervals cannot do
this. This is the zonotope domain of DeepZ (Singh et al., NeurIPS 2018), applied to two networks at once.

## 5. Differential analysis: the layer rules  (analysis.py, activations.py)

The analysis follows `h` (the original network's values) and `d = h' - h` (compiled minus original).

### 5.1 Affine layer

With `z = W h + b` and `z' = W' h' + b'` and `h' = h + d`:

```
delta = z' - z = W' d + (W' - W) h + (b' - b)
```

This is an identity, so it is exact for forms. `W' - W` and `b' - b` are computed in float64, and their
rounding (at most `2u` times the result) becomes the weight and bias radius of the difference op.

### 5.2 ReLU and LeakyReLU, `sigma(z) = z` for `z >= 0` and `a z` otherwise (`a = 0` for ReLU)

Given forms for `z` (original pre-activation) and `delta`, and the form `z' = z + delta`, take their
ranges `[zl, zu]`, `[dl, du]`, `[z'l, z'u]` by concretization.

**Original values `h = sigma(z)`.**

* `zl >= 0`: `h = z`. `zu <= 0`: `h = a z`. Both exact.
* `zl < 0 < zu` (the neuron can switch inside X): for any `lam` in `[a, 1]`, `g(z) = sigma(z) - lam z` is
  `(a - lam) z >= 0` for `z <= 0` and `(1 - lam) z >= 0` for `z >= 0`, so on `[zl, zu]` it ranges over
  `[0, max(g(zl), g(zu))]`. With `M` that maximum, `h = lam z + M/2 + (M/2) eps_new`. We use the chord slope
  `lam = (zu - a zl) / (zu - zl)` (DeepZ's choice for `a = 0`); soundness holds for any computed `lam`
  in `[a, 1]`, so the float64 value of `lam` needs no error analysis.

**Difference `d_new = sigma(z + delta) - sigma(z)`.** When both networks have the neuron stably on or off,
`sigma` is linear on each side and the difference is exact:

| original `z` | compiled `z'` | `d_new` |
|---|---|---|
| on (`zl >= 0`) | on (`z'l >= 0`) | `delta` |
| off (`zu <= 0`) | off (`z'u <= 0`) | `a delta` |
| on | off | `a z' - z` |
| off | on | `z' - a z` |

Otherwise we enclose `F(z, delta) = sigma(z + delta) - sigma(z)` over the box `[zl, zu] x [dl, du]`:

* `F` is nondecreasing in `delta`, because `sigma` is nondecreasing.
* For fixed `delta >= 0`, `F` is nondecreasing in `z`; for fixed `delta <= 0`, it is nonincreasing. The reason
  is that `sigma` is convex (its slope `a <= 1` jumps up to 1), so `sigma'(z + delta) - sigma'(z)` has the
  sign of `delta`.
* Hence `min F = min(F(zl, dl), F(zu, dl))` and `max F = max(F(zl, du), F(zu, du))`: the four corners.

This box is intersected with the independent enclosure `[sigma(z'l) - sigma(zu), sigma(z'u) - sigma(zl)]`
and becomes a fresh symbol. If `delta` is exactly zero, `d_new` is exactly zero. This is what makes two
identical models get a bound of exactly 0 when their weights are used as stored. With BatchNorm, each
model's folded weights carry their own rounding radius, which leaves a remainder of about `1e-15`.

Worked example (`tests/test_hand_cases.py`): `relu(x)` against `relu(1.5 x)` on `[-1, 1]`. Here `z` is in
`[-1, 1]` and `delta = 0.5 x` is in `[-0.5, 0.5]`. The corners give `F` in `[-0.5, 0.5]`, so the bound is 0.5,
which is the true maximum at `x = 1`. Relaxing each network separately gives 1.0; intervals give 1.5.

### 5.3 tanh and sigmoid

Both derivatives are even and decrease in `|x|`. We compute them only from `exp`, in forms that cannot
overflow: `tanh'(x) = 4 t / (1 + t)^2` with `t = exp(-2|x|)`, and `sigmoid'(x) = t / (1 + t)^2` with
`t = exp(-|x|)`. A slope computed below `2^-299` (where `exp` may have underflowed) is replaced by the
enclosure `[0, 2^-298]`. Over an interval, the largest slope is at the point closest to 0 and the smallest
at an endpoint. `tests/test_review_regressions.py` checks these bounds out to `|x| = 800` against
70-digit decimal arithmetic.

* `h = sigma(z)`: take `lam` = a lower bound on `sigma'` over `[zl, zu]`. Then `sigma(z) - lam z` is
  nondecreasing there, so it ranges over `[sigma(zl) - lam zl, sigma(zu) - lam zu]`, which gives `mid` and `rad`.
* `d_new`: by the mean value theorem, `sigma(z + delta) - sigma(z) = sigma'(xi) delta` for some `xi` between `z`
  and `z + delta`, so `xi` lies in `hull([zl, zu], [z'l, z'u])`. With `[m, M]` bounding `sigma'` there and
  `kappa = (m + M) / 2`:
  `d_new = kappa delta + (sigma'(xi) - kappa) delta`, and the second term is at most `((M - m)/2) max|delta|`.
  This keeps the dependence on `delta`. We also compute the corner enclosure `[m, M] x [dl, du]`, intersected
  with the independent one, and keep whichever is narrower per neuron. Both are sound.

The float64 `tanh`, `sigmoid` and `exp` of PyTorch are assumed accurate to relative `2^-45`; the test
`test_float64_library_functions_meet_the_assumption` checks this against 50-digit decimal arithmetic.
Their real error is close to `2^-52`.

## 6. From forms to the output bound

### 6.1 Without softmax  (api.py)

`R_i = |c_i| + sum_k |G_k,i| + e_i` for the form of `d` at the output, rounded up.

### 6.2 Softmax  (softmax.py)

Let `p = softmax(v)`. The Jacobian is `J_ij = p_i (1[i=j] - p_j)`. Two facts:

1. Row `i` of `J` has absolute sum `p_i (1 - p_i) + p_i sum_{j != i} p_j = 2 p_i (1 - p_i) <= 1/2`.
2. Every row of `J` sums to zero, so `J (delta - c 1) = J delta` for every constant `c`. Adding the same
   number to all logits does not change softmax.

With `v' = v + delta` and `p(t) = softmax(v + t delta)`, the mean value theorem in integral form gives

```
|p_i(v') - p_i(v)|  <=  max_t 2 p_i(t)(1 - p_i(t))  *  min_c || delta - c 1 ||_inf
                    =   max_t p_i(t)(1 - p_i(t))  *  (max_j delta_j - min_j delta_j)
```

* `max_j delta_j - min_j delta_j` is bounded by the largest upper end of the pairwise forms `d_j - d_k`. They
  are formed from the forms, so shared terms cancel.
* `p_i(t) = 1 / (1 + sum_{j != i} exp(v_j(t) - v_i(t)))`, and `v_j(t) - v_i(t)` lies between the
  corresponding differences of the two networks. The pairwise forms of each network's logits therefore give
  a range `[pl_i, pu_i]` for every point of the segment.
* `p(1 - p)` increases up to `1/2` and decreases after, so its maximum over `[pl_i, pu_i]` is `1/4` if the
  interval contains `1/2`, else its value at the endpoint closer to `1/2`.

For a confident classifier `p_i (1 - p_i)` is tiny, which makes this far tighter than the global constant
`1/2`. A hand-checkable case (`tests/test_softmax_and_certificate.py`): logits `(0, 0)` against
`(eta, -eta)`. The true difference is `0.5 tanh(eta)`; the bound is `(1/4)(2 eta) = eta / 2`.

### 6.3 Prediction certificate  (api.py)

The certificate is about the bounded outputs.

* **Without softmax:** if, for some class `c` and every `j != c`, the lower end of the form `v_c - v_j`
  exceeds the two execution allowances `A_c + A_j`, then both executed models rank `c` strictly first for
  every input in X.
* **With softmax:** we compare the executed probability ranges `[pl_c - A_c, pu_c + A_c]` directly and require
  `pl_c - A_c > pu_j + A_j`. A logit ordering is not enough, because rounding can map two different logits to
  equal probabilities. Logits `0` and `1e-8` both give probability `0.5` in float32, and the argmax then
  breaks the tie by position.

## 7. Our own float64 arithmetic  (rounding.py, intervals.py)

Round-to-nearest makes every single operation exact up to half a step to the next float, so:

* a lower end is moved one float down (`down`) and an upper end one float up (`up`) after it is computed;
* a nonnegative quantity computed as a sum of at most `n` products of nonnegative numbers is at least
  `(1 - gamma_n)` times its exact value, so `inflate` multiplies by `(1 + 2 gamma_{n+2})` and steps up one float
  (`deflate` is the lower-bound version).

**Exact zeros are kept.** A rounded sum is zero only if the exact sum is zero (sums that land in the
subnormal range are exact). A rounded product is zero only if a factor is zero, provided no product
underflows. `check_range` stops the analysis if any nonzero value falls outside `[2^-400, 2^400]`, so no
product of two accepted values can underflow or overflow.

`test_rounding_intervals.py` compares these operations with exact rational arithmetic (`fractions.Fraction`).

## 8. Floating-point execution of the models  (fp_execution.py)

For one network, `a_k` bounds `|executed value - real value|` after op `k`, over all of X. Value sizes come
from a zonotope analysis of that network alone: `|x| <= mag(range of x) + a`.

* **Affine** `y = W x + b` with `n` inputs:
  `a_new = |W| a + gamma_{n+1} (|W| |x| + |b|) + (n + 1) 2^-126`.
  The first term carries the error that arrived. The second is the rounding of the layer's own dot products,
  for any summation order and with or without fused multiply-add (A1). The last covers underflow, including
  flush-to-zero.
* **Eval-mode BatchNorm** executed as its own layer, as `x * alpha + (beta - mean * alpha)` with
  `alpha = w / sqrt(var + eps)` (A3). The real scale is `s = w / sqrt(D)` with `D = var + eps`.
  1. *Denominator.* PyTorch may use `eps` exactly or after rounding it to the model's format (`eps*`), and
     `var + eps` is rounded once more. So the executed denominator `d` lies in
     `[S_lo (1 - u) - 2^-126, S_hi (1 + u) + 2^-126]`, with `S` taken over both choices of `eps`. If this range
     reaches 0 we refuse the model: for example `eps = 1e-44` is subnormal in float32 and is flushed to 0
     under flush-to-zero. Negative running variances are refused at extraction.
  2. *Scale.* `alpha / s = sqrt(D / d)` times at most 4 roundings, so `|alpha - s| <= e |s| + 2^-126` with
     `e = r + gamma_4 (1 + r)` and `r = max |sqrt(D / d) - 1|` over the range of `d`.
  3. *Output.* `y^ - y = (alpha - s)(x - mean)` plus the roundings of `mean * alpha`, the subtraction,
     `x * alpha` and the sum. With `|alpha| <= (1 + e)|s| + 2^-126`:

     `a_new = |alpha| a + e |s| (|x| + |mean|) + gamma_4 (|alpha| (|x| + |mean|) + |beta|) + 2^-126 (|x| + |mean| + 3)`.

  The last term covers underflow of `alpha` (for example `1e-25 / sqrt(1e38)` is subnormal), whose absolute
  error is then multiplied by `x` and `mean`.
* **ReLU**: `max(x, 0)` is exact in floating point and 1-Lipschitz, so `a_new = a`.
  **LeakyReLU**: PyTorch first rounds the Python slope `a` to the model's format (`a*`), then multiplies, so
  `a_new = a + (u a* + |a* - a|) |x| + 2^-126`.
* **tanh, sigmoid**: `a_new = L a + K |sigma| + 2^-126`, with `L = 1` (tanh) or `1/4` (sigmoid), and `K` the
  library accuracy of A2 for the model's format (`16 u` in float32, `2^-45` in float64).
* **Final softmax** with incoming error `a` (vector over the logits):
  * *Carried:* `|p_i(v^) - p_i(v~)| <= 2 max p_i(1 - p_i) * max_j a_j` (section 6.2, fact 1). The range of
    `p` is taken over all logits within `a` of the real ones.
  * *Own rounding (A4):* the executed softmax computes `m = max_j v_j`, `t_j = fl(v_j - m)`,
    `e_j = fl(exp(t_j))`, `s = fl(sum e_j)` and `p_i = fl(e_i / s)` (or `e_i * fl(1/s)`). Write
    `D >= max_j v_j - min_j v_j` for the executed logits. Then:
    * `t_j = (v_j - m)(1 + theta)` with `|theta| <= u`, so `exp(t_j) = exp(v_j - m) exp(phi_j)` with `|phi_j| <= D u`;
    * `e_j = exp(t_j)(1 + theta_2)` with `|theta_2| <= K` (A2, the format's library accuracy);
    * `s = sum_j e_j (1 + theta_3,j)` with `|theta_3,j| <= gamma_{n-1}`;
    * the division costs at most two more factors `(1 + theta)` with `|theta| <= u`.

    So `p^_i / p_i` is the product of the numerator's factors and the division's factors, divided by a weighted
    mean (weights `p_j`) of the denominator's factors. Every factor lies in `[exp(-1.01 alpha), exp(1.01 alpha)]`
    with `alpha` its first-order size, because `1/(1 - alpha) <= exp(1.01 alpha)` and
    `1 - alpha >= exp(-1.01 alpha)` for `alpha <= 0.01`. The sizes add up to
    `S <= (2D + 2) u + 2K + gamma_{n-1} <= kappa`, where `kappa = (2D + 6) u + 2K + gamma_n`. Hence

    ```
    |p^_i / p_i - 1| <= exp(1.01 S) - 1 <= 1.01 S (1 + 1.01 S) <= 1.03 kappa     (when kappa <= 0.01)
    ```

    and `|p^_i - p_i| <= 1.03 kappa p_i + 2^-126`. The code refuses to run if `kappa > 0.01`.
  * The allowance is the sum of the carried and the own term.

The test `test_allowance_covers_observed_float32_rounding` compares float32 and float64 executions over
thousands of inputs, for every activation, with and without softmax, plus BatchNorm and convolutions.

## 9. Why the whole bound is sound

Assumptions:

* **A1** Both models run on the CPU in IEEE float32 (or float64) with round-to-nearest. Matrix products and
  convolutions use full precision (no TF32 or bfloat16), in any order, with or without fused multiply-add,
  and every operation keeps the model's dtype. Enforced:
  * `check_runtime` refuses a float32 matmul precision other than "highest", any `fp32_precision` switch
    (global, oneDNN matmul/conv/rnn) other than IEEE, active CPU autocast, and parameters or buffers off the CPU;
  * extraction refuses any node whose output dtype differs from the model's.
* **A2** float32 `tanh`, `sigmoid` and `exp` are accurate to `16 u`; float64 ones to `2^-45`. Both are tested
  on grids; the allowance uses the constant of each model's format.
* **A3** Eval-mode BatchNorm executes as `x * alpha + (beta - mean * alpha)`, `alpha = w / sqrt(var + eps)`,
  each step rounded once, with `eps` used exactly or rounded to the model's format. Underflow and the
  denominator are handled as in section 8. Enforced: negative running variance and denominators that could
  round to zero are refused.
* **A4** Softmax executes as `exp(x - max) / sum`. We read this in the C++ source Inductor generates for
  the MNIST model. For eager PyTorch it is checked only empirically, by the tests that compare float32
  and float64 executions.
* **A5** No overflow. Checked: the analysis stops if a value could exceed a quarter of the format's maximum.
* **A6** Each model computes exactly the chain traced by `torch.fx`. A model from `torch.compile` computes
  the same real-number function as its original module: Inductor fuses and reorders operations, and on
  Windows it compiles without fast-math flags (`torch/_inductor/cpp_builder.py`, torch 2.14). `check_runtime`
  refuses runs with Inductor's unsafe-math or fast-math switches turned on, read from the live config
  (`cpp.enable_unsafe_math_opt_flag`, `use_fast_math`) and from the environment.

The argument:

1. `X` is inside the analysed box, because domains are rounded outward (`domain.py`).
2. By induction over the ops, the invariant of section 4 holds for `h` and `d` after every op (sections 4
   and 5), including our float64 rounding (section 7). So at the output, every true `d_i(x)` lies in the
   concretization of its form, and `R_i` bounds `|g~_i(x) - f~_i(x)|` for every `x` in X. With a final
   softmax, section 6.2 turns the forms into a bound on the probabilities.
3. Under A1 to A5, `A_f` and `A_g` bound the execution error of each model over X (section 8).
4. The triangle inequality (section 2) combines the three parts. Taking the minimum over analyses is
   sound, because each analysis is.

Sampling and the gradient attack (`empirical.py`) are not part of the argument. They can refute a bound,
never confirm one; they are the sanity check the task asks for.

## 10. Where the bound is exact and where it loses precision

**Exact (all tested):**
* Two identical models whose weights are used as stored: `R = 0`. With BatchNorm folding, each model's
  folded weights carry their own float64 rounding radius, which leaves about `1e-15`.
* A single input point, for outputs without a final softmax: `R` equals the true real difference, up to
  float64 slack. With a final softmax the lemma of section 6.2 is not exact (for logits `(0, 0)` against
  `(1, -1)` it gives 0.5 for a true difference of 0.38).
* A ReLU network in which no neuron can switch inside X: both networks are affine on X, and `R` equals
  `max |dA c + dc| + |dA| r`.

**Precision is lost:**
* At neurons that can switch inside X: a fresh symbol replaces an exact form. On the MNIST model at
  `eps = 0.01` the logit bound is about 2.1 times the largest difference found.
* In the rounding allowance. The float32 bound assumes every rounding error lines up in the worst way.
  At the logits of the MNIST model it is about 0.22. For `torch.compile`, whose difference from the
  original is pure rounding, the bound (at least `1.3e-5` on the probabilities) is several hundred times
  the largest deviation observed (`3.0e-8`).
* In the interval methods. They forget correlations, which is why they are kept only as baselines and
  cross-checks.

Interval arithmetic is inclusion-monotone: a larger X never gives a smaller interval bound. The zonotope
relaxations do not promise this, because `lam` depends on the ranges, and the tests check monotonicity only
for the interval method.

## 11. Cost

For a chain with input dimension `k`, widths `n_l`, and `s` fresh symbols, each affine layer multiplies
a `(k + s) x n_in` generator matrix by an `n_in x n_out` matrix. For the 784-256-256-10 MNIST model one
call (four analyses plus two rounding passes) takes a median of about 0.2 seconds on the CPU used for the
results in `results/`. Memory grows with `(k + s)` times the layer size, which is the limit for large
convolutional networks.

## References

* N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. Section 3.1 (error
  of inner products, the constant gamma_n).
* G. Singh, T. Gehr, M. Mirman, M. Puschel, M. Vechev. Fast and Effective Robustness Certification.
  NeurIPS 2018 (DeepZ: the zonotope domain and its ReLU transformer).
* B. Paulsen, J. Wang, C. Wang. ReluDiff: Differential Verification of Deep Neural Networks. ICSE 2020,
  arXiv:2001.03662 (differential analysis of two networks).
* B. Paulsen, J. Wang, J. Wang, C. Wang. NeuroDiff: Scalable Differential Verification of Neural
  Networks using Fine-Grained Approximation. ASE 2020, arXiv:2009.09943.
* D. Zombori, B. Banhelyi, T. Csendes, I. Megyeri, M. Jelasity. Fooling a Complete Neural Network Verifier.
  ICLR 2021 (verifiers that ignore floating point can be fooled).
* K. Jia, M. Rinard. Exploiting Verified Neural Networks via Floating Point Numerical Error. SAS 2021.
