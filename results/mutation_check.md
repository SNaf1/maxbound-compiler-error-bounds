# Planted-bug check

18 of 18 planted soundness bugs are caught by the test suite (`python tools/mutation_check.py`).

| bug | what it does | caught | first failing test |
|---|---|---|---|
| `drop-weight-change` | difference at a linear layer forgets (W' - W) h + (b' - b) | yes | tests/test_exactness.py::test_bound_equals_true_maximum_when_no_relu_can_switch |
| `switching-relu-as-on` | a ReLU that can switch inside X is treated as always on | yes | tests/test_hand_cases.py::test_relu_case |
| `on-off-sign` | wrong sign when the neuron is on in the original and off in the compiled model | yes | tests/test_exactness.py::test_zero_width_box_reproduces_the_forward_pass[relu] |
| `off-on-formula` | wrong formula when the neuron is off in the original and on in the compiled model | yes | tests/test_exactness.py::test_zero_width_box_reproduces_the_forward_pass[relu] |
| `wrong-corners` | four-corner rule takes the minimum at the wrong corners | yes | tests/test_activations.py::test_difference_range_contains_sampled_values[relu] |
| `thin-relu-band` | ReLU relaxation band half as wide as needed | yes | tests/test_activations.py::test_value_range_and_relaxation_contain_the_function[relu] |
| `deriv-at-midpoint` | largest tanh/sigmoid slope taken at the midpoint, not the point closest to 0 | yes | tests/test_activations.py::test_derivative_bounds_hold_on_a_fine_grid[tanh] |
| `forget-interval-part` | concretization ignores the interval part e of a form | yes | tests/test_exactness.py::test_bound_equals_true_maximum_when_no_relu_can_switch |
| `no-float64-slack` | our own float64 rounding in linear layers is ignored | yes | tests/test_rounding_intervals.py::test_affine_layer_encloses_the_exact_rational_result |
| `no-outward-rounding` | endpoints are not rounded outward | yes | tests/test_rounding_intervals.py::test_up_and_down_move_one_float_and_keep_exact_zero |
| `softmax-half` | softmax bound off by a factor of 2 | yes | tests/test_softmax_and_certificate.py::test_softmax_bound_is_tight_for_a_symmetric_shift |
| `bn-fold-no-shift` | BatchNorm folding drops the shift t | yes | tests/test_graph.py::test_batchnorm_fusion_is_verified_not_assumed |
| `no-dot-rounding` | float32 rounding of the models' own dot products is ignored | yes | tests/test_fp_execution.py::test_allowance_covers_observed_float32_rounding[False-relu] |
| `no-softmax-rounding` | float32 rounding inside softmax is ignored | yes | tests/test_fp_execution.py::test_allowance_covers_softmax_rounding_on_its_own |
| `bn-no-underflow` | underflow of BatchNorm's precomputed scale is ignored | yes | tests/test_review_regressions.py::test_batchnorm_with_a_subnormal_scale[True] |
| `leaky-slope-exact` | LeakyReLU's slope is assumed to be stored exactly in float32 | yes | tests/test_review_regressions.py::test_leaky_relu_slope_is_rounded_to_float32_before_use |
| `domain-via-float32` | Python lists are converted to float32 before float64 | yes | tests/test_review_regressions.py::test_list_domains_are_kept_in_float64 |
| `certificate-no-allowance` | the softmax certificate ignores the rounding allowance | yes | tests/test_review_regressions.py::test_certificate_is_refused_when_probabilities_can_tie |
