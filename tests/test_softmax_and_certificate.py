"""Softmax outputs and the prediction-agreement certificate."""
import torch
import torch.nn as nn

from maxbound import LinfBall, MaxBound
from maxbound.compilers import quantize_weights
from maxbound.empirical import sampled_max_difference


def _linear_classifier(bias, softmax=False):
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    with torch.no_grad():
        lin.weight.mul_(0.1)
        lin.bias.copy_(torch.tensor(bias))
    return (nn.Sequential(lin, nn.Softmax(dim=-1)) if softmax else nn.Sequential(lin)).eval()


def test_softmax_bound_is_tight_for_a_symmetric_shift():
    """Logits (0, 0) against (eta, -eta), whatever the input. The probabilities
    are (1/2, 1/2) and (sigmoid(2 eta), sigmoid(-2 eta)), so the true difference
    is sigmoid(2 eta) - 1/2 = tanh(eta) / 2. The bound is max p(1-p) * spread =
    (1/4) * (2 eta) = eta / 2, tight to first order in eta."""
    eta = 1e-3
    a = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1)).eval()
    b = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=-1)).eval()
    with torch.no_grad():
        for m, bias in ((a, [0.0, 0.0]), (b, [eta, -eta])):
            m[0].weight.zero_()
            m[0].bias.copy_(torch.tensor(bias, dtype=torch.float64))
    eta32 = float(torch.tensor(eta, dtype=torch.float32))
    true = 0.5 * float(torch.tanh(torch.tensor(eta32, dtype=torch.float64)))
    bound = MaxBound(a, b, LinfBall(torch.zeros(2), 1.0), fp_model="real")
    assert true <= bound.real_bound <= true * (1 + 1e-6)


def test_certificate_for_a_confident_model():
    m = _linear_classifier([10.0, 0.0, 0.0])
    bound = MaxBound(m, quantize_weights(m, 8), LinfBall(torch.zeros(4), 0.1))
    assert bound.same_prediction == 0


def test_no_certificate_when_classes_tie():
    m = _linear_classifier([0.0, 0.0, 0.0])
    bound = MaxBound(m, quantize_weights(m, 8), LinfBall(torch.zeros(4), 0.1))
    assert bound.same_prediction is None


def test_softmax_bound_uses_the_confidence():
    """A confident classifier has p(1 - p) far below 1/4, so the probability bound
    is much smaller than half the logit bound (the global softmax constant)."""
    m = _linear_classifier([8.0, 0.0, 0.0], softmax=True)
    cl = quantize_weights(m, 6)
    X = LinfBall(torch.zeros(4), 0.5)
    probs = MaxBound(m, cl, X, fp_model="real")
    logits = MaxBound(m, cl, X, fp_model="real", output="logits")
    assert probs.output_kind == "probabilities" and logits.output_kind == "outputs"
    assert probs.real_bound < 0.05 * logits.real_bound
    seen, _ = sampled_max_difference(m, cl, X, n=2000)
    assert seen <= float(MaxBound(m, cl, X))
