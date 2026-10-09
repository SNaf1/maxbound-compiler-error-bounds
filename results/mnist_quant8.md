# dacorvo/mnist-mlp, 8-bit per-channel weight quantization

Model `dacorvo/mnist-mlp` @ `7ed1e96d182395fd7380ea8495b9eb2364ce9dc1`, compiler `quantize_weights` {'bits': 8, 'per_channel': True}, 50 test images. Medians over images; eps is in pixel units (pixels in [0, 1]).

| output | eps | violations | same class certified | bound | real part | fp allowance | largest found | real / found | zonotope | interval | zonotope-sep. | interval-sep. | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probabilities | 0 | 0/50 | 50/50 | 2.05e-05 | 4.32e-07 | 2e-05 | 1.19e-07 | 3.04 | 4.32e-07 | 4.32e-07 | 4.32e-07 | 4.32e-07 | 0.18 |
| probabilities | 0.001 | 0/50 | 49/50 | 2.11e-05 | 5.57e-07 | 2.05e-05 | 2.56e-07 | 2.40 | 5.57e-07 | 0.000155 | 7.02e-07 | 0.00836 | 0.18 |
| probabilities | 0.003 | 0/50 | 49/50 | 2.36e-05 | 9.81e-07 | 2.23e-05 | 3.58e-07 | 2.55 | 9.81e-07 | 0.0849 | 4.15e-06 | 6.42 | 0.18 |
| probabilities | 0.01 | 0/50 | 49/50 | 6.69e-05 | 1.19e-05 | 5.49e-05 | 5.96e-07 | 10.34 | 1.19e-05 | 0.31 | 0.000167 | 21.4 | 0.20 |
| logits | 0 | 0/50 | 50/50 | 0.229 | 0.0278 | 0.218 | 0.0278 | 1.00 | 0.0278 | 0.0278 | 0.0278 | 0.0278 | 0.16 |
| logits | 0.001 | 0/50 | 49/50 | 0.232 | 0.0295 | 0.219 | 0.0282 | 1.03 | 0.0295 | 0.0773 | 0.0347 | 4.58 | 0.17 |
| logits | 0.003 | 0/50 | 49/50 | 0.239 | 0.0354 | 0.219 | 0.029 | 1.17 | 0.0354 | 0.187 | 0.0924 | 13.7 | 0.17 |
| logits | 0.01 | 0/50 | 49/50 | 0.28 | 0.07 | 0.221 | 0.0324 | 2.12 | 0.07 | 0.642 | 0.879 | 44.2 | 0.19 |
