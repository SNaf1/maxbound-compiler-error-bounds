"""Quickstart: bound how much 8-bit weight quantization can change a Hugging Face
MNIST classifier, for every input within eps of a real test image.

    pip install -e ".[hf]"
    python examples/quickstart.py

The model is dacorvo/mnist-mlp (Linear-ReLU-Linear-ReLU-Linear-softmax). Its
model file is fetched from the Hub and run with trust_remote_code; it is a short
file that only defines the three layers (read it at
https://huggingface.co/dacorvo/mnist-mlp/blob/main/modeling_mlp.py).
"""
import io
import logging

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModel

from maxbound import LinfBall, MaxBound
from maxbound.compilers import quantize_weights
from maxbound.empirical import attack_max_difference, sampled_max_difference

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")   # show MaxBound's steps

MODEL, MODEL_REV = "dacorvo/mnist-mlp", "7ed1e96d182395fd7380ea8495b9eb2364ce9dc1"
DATA, DATA_REV = "ylecun/mnist", "77f3279092a1c1579b2250db8eafed0ad422088c"
MEAN, STD = 0.1307, 0.3081                 # the model card's normalisation

# 1. the original model, and the "compiler": 8-bit weight quantization
model = AutoModel.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=True).eval()
cl_func = lambda m: quantize_weights(m, bits=8)

# 2. the input domain X: every image within eps (in pixel units) of test image 0,
#    kept inside the valid pixel range [0, 1], expressed in the model's normalised units
table = pq.read_table(hf_hub_download(DATA, "mnist/test-00000-of-00001.parquet", repo_type="dataset", revision=DATA_REV))
image = table.column("image")[0].as_py()
pixels = torch.from_numpy(np.asarray(Image.open(io.BytesIO(image["bytes"])), dtype=np.float32).reshape(-1) / 255.0)
eps = 0.003
lo, hi = (float(v) for v in (torch.tensor([0.0, 1.0]) - MEAN) / STD)   # pixel range, same float32 arithmetic
X = LinfBall((pixels - MEAN) / STD, eps / STD, clip=(lo, hi))

# 3. the snippet from the task, as written (L-infinity over the 10 output probabilities)
cl_model = cl_func(model)
bound = MaxBound(model, cl_model, X)
print(f"MaxBound = {float(bound):.3g}")
print(bound.summary())

# 4. sanity check: sampling and a gradient attack can only find smaller differences
sampled, _ = sampled_max_difference(model, cl_model, X, n=2000)
attacked, _ = attack_max_difference(model, cl_model, X, steps=100)
print(f"largest difference found: sampling {sampled:.3g}, attack {attacked:.3g}")
assert max(sampled, attacked) <= float(bound)
