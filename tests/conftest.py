"""Small models and helpers shared by the tests."""
import copy

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def make_mlp(widths, act="relu", seed=0, softmax=False):
    torch.manual_seed(seed)
    layers = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            layers.append(ACTIVATIONS[act]())
    if softmax:
        layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers).eval()


def perturb(model, seed=1, size=0.02):
    """Copy of the model with Gaussian noise of std ``size`` added to every parameter."""
    m = copy.deepcopy(model).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(size * torch.randn(p.shape, generator=g))
    return m


def randomize_batchnorm(model, seed=0):
    """Give every BatchNorm non-trivial running statistics and affine parameters."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.running_mean.copy_(torch.randn(m.num_features, generator=g))
                m.running_var.copy_(0.5 + torch.rand(m.num_features, generator=g))
                m.weight.copy_(1.0 + 0.3 * torch.randn(m.num_features, generator=g))
                m.bias.copy_(0.3 * torch.randn(m.num_features, generator=g))
    return model.eval()


def small_cnn(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2d(4, 4, 3, stride=2), nn.ReLU(),
        nn.Flatten(), nn.Linear(4 * 3 * 3, 3),
    )
    return randomize_batchnorm(model, seed)
