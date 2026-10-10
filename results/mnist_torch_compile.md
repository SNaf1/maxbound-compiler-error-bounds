# dacorvo/mnist-mlp, torch.compile (Inductor, CPU)

Model `dacorvo/mnist-mlp` @ `7ed1e96d182395fd7380ea8495b9eb2364ce9dc1`, compiler `torch_compile` {}, 50 test images. Medians over images; eps is in pixel units (pixels in [0, 1]).

| output | eps | violations | same class certified | bound | real part | fp allowance | largest found | real / found | zonotope | interval | zonotope-sep. | interval-sep. | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probabilities | 0 | 0/50 | 50/50 | 2e-05 | 0 | 2e-05 | 0 | inf | 0 | 0 | 3.26e-15 | 3.27e-15 | 0.17 |
| probabilities | 0.01 | 0/50 | 49/50 | 5.49e-05 | 0 | 5.49e-05 | 9.09e-13 | 0.00 | 0 | 0 | 0.000162 | 21.4 | 0.18 |
| logits | 0 | 0/50 | 50/50 | 0.218 | 0 | 0.218 | 0 | inf | 0 | 0 | 1.48e-10 | 1.49e-10 | 0.16 |
| logits | 0.01 | 0/50 | 49/50 | 0.221 | 0 | 0.221 | 0 | inf | 0 | 0 | 0.843 | 44.2 | 0.17 |
