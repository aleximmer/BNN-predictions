import torch
import numpy as np
import pytest

from preds.likelihoods import GaussianLh, CategoricalLh


@pytest.fixture
def sigma_noise():
    return 17.7


@pytest.fixture
def f(sigma_noise):
    torch.manual_seed(89)
    f = sigma_noise * torch.randn(10, 1)
    return f


@pytest.fixture
def y(sigma_noise):
    torch.manual_seed(77)
    y = sigma_noise * torch.randn(10, 1)
    return y


@pytest.fixture
def fcat():
    torch.manual_seed(89)
    f = torch.randn(10, 2)
    return f


@pytest.fixture
def ycat():
    torch.manual_seed(89)
    y = torch.randint(2, (10,))
    return y


def test_gaussian_residual(f, y, sigma_noise):
    # gaussian residual is (y - f) / sigma_noise^2
    sigma_var = sigma_noise ** 2
    lh = GaussianLh(sigma_noise)
    assert torch.allclose(lh.residual(y, f), (y - f) / sigma_var)


def test_gaussian_hess(f, sigma_noise):
    # hessian is 1 / sigma^2
    sigma_var = sigma_noise ** 2
    lh = GaussianLh(sigma_noise)
    hess = lh.Hessian(f)
    assert torch.allclose(hess, torch.ones_like(hess) / sigma_var)
    assert hess.shape[1] == hess.shape[2]


def test_gaussian_link(f, sigma_noise):
    lh = GaussianLh(sigma_noise)
    assert torch.equal(lh.inv_link(f), f)


def test_gaussian_nn_loss(f, y, sigma_noise):
    # make sure gradients are equivalent of
    # nn_loss and negative log likelihood!
    lh = GaussianLh(sigma_noise)
    nn_loss, factor = lh.nn_loss()
    f.requires_grad = True
    log_lik = -lh.log_likelihood(y, f)
    log_lik.backward()
    grad_f = f.grad.data.numpy().copy()
    f.grad.data.zero_()
    loss = factor * nn_loss(f, y)
    loss.backward()
    grad_floss = f.grad.data.numpy().copy()
    assert np.abs(grad_f - grad_floss).max() < 1e-7


def test_categorical_residual(fcat, ycat):
    # categorical residual is ycat - softmax(f)
    # also tests invlink implicitly!
    lh = CategoricalLh()
    softmax = torch.softmax(fcat, dim=-1)
    labels = torch.zeros_like(softmax)
    labels[torch.arange(0, len(labels)), ycat] = 1.
    residual_true = labels - softmax
    assert torch.allclose(residual_true, lh.residual(ycat, fcat))


def test_categorical_hessian(fcat):
    # cannot test, algebraic implementation.
    # make sure it runs and is decomposable!
    lh = CategoricalLh()
    H = lh.Hessian(fcat)
    ls = torch.symeig(H)[0]
    assert ls.shape[0] == len(fcat) and ls.shape[1] == len(fcat[0])
    assert H.shape[1] == H.shape[2]


def test_categorical_nn_loss(fcat, ycat):
    lh = CategoricalLh()
    nn_loss, factor = lh.nn_loss()
    fcat.requires_grad = True
    log_lik = -lh.log_likelihood(ycat, fcat)
    log_lik.backward()
    grad_f = fcat.grad.data.numpy().copy()
    fcat.grad.data.zero_()
    loss = factor * nn_loss(fcat, ycat)
    loss.backward()
    grad_floss = fcat.grad.data.numpy().copy()
    assert np.abs(grad_f - grad_floss).max() < 1e-6
