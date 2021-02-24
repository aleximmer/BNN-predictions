import numpy as np
import torch
from preds.likelihoods import BernoulliLh, Bernoulli, CategoricalLh, Categorical


def acc(g, y, likelihood=None):
    """Binary accuracy"""
    if type(likelihood) is CategoricalLh:
        return macc(g, y)
    y_pred = (g >= 0.5).type(y.dtype)
    return torch.sum((y_pred == y).float()).item() / len(y_pred)


def macc(g, y):
    """Multiclass accuracy"""
    return torch.sum(torch.argmax(g, axis=-1) == y).item() / len(y)


def nll_cls(p, y, likelihood):
    """Avg. Negative log likelihood for classification"""
    if type(likelihood) is BernoulliLh:
        p_dist = Bernoulli(probs=p)
        return - p_dist.log_prob(y).mean().item()
    elif type(likelihood) is CategoricalLh:
        p_dist = Categorical(probs=p)
        return - p_dist.log_prob(y).mean().item()
    else:
        raise ValueError('Only Bernoulli and Categorical likelihood.')


def ece(probs, labels, likelihood=None, bins=10):
    # source: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    if type(likelihood) is BernoulliLh:
        probs = torch.stack([1-probs, probs]).t()
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels.long())

    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def setup_grid(X, h=0.01, buffer=1):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - buffer, X[:, 0].max() + buffer
    y_min, y_max = X[:, 1].min() - buffer, X[:, 1].max() + buffer
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()])
    return grid, xx, yy


def kron_ggn(model, stochastic):
    if stochastic:  # KFAC
        return [p.kfac for p in model.parameters()]
    else:  # KFLR
        return [p.kflr for p in model.parameters()]


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def get_sym_psd(dim=3):
    x = torch.randn(dim, dim*3)
    M = x @ x.T
    return M
