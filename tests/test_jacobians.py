import pytest
import torch

from preds.models import CIFAR10Net
from preds.models import MLPS
from preds.gradients import Jacobians, Jacobians_naive


@pytest.fixture
def model():
    torch.manual_seed(71)
    model = CIFAR10Net(in_channels=2, n_out=3)
    return model


@pytest.fixture
def linear_model():
    torch.manual_seed(77)
    model = MLPS(28*28*2, [], output_size=1, flatten=True, bias=False)
    return model


@pytest.fixture
def X():
    torch.manual_seed(15)
    batch_size = 20
    channels = 2
    pixels = 28
    X = torch.randn(batch_size, channels, pixels, pixels)
    return X


@pytest.fixture
def y():
    torch.manual_seed(15)
    batch_size = 20
    classes = 3
    y = torch.softmax(torch.randn(batch_size, classes), dim=1).argmax(dim=1)
    return y


def test_naive_jacobians(linear_model, X):
    # jacobian of linear model is input X.
    Js, f = Jacobians_naive(linear_model, X)
    true_Js = X.reshape(len(X), -1)
    assert true_Js.shape == Js.shape
    assert torch.allclose(true_Js, Js)
    assert torch.allclose(f, linear_model(X))


def test_linear_jacobians(linear_model, X):
    # jacobian of linear model is input X.
    Js, f = Jacobians(linear_model, X)
    true_Js = X.reshape(len(X), -1)
    assert true_Js.shape == Js.shape
    assert torch.allclose(true_Js, Js, atol=1e-5)
    assert torch.allclose(f, linear_model(X), atol=1e-5)


def test_jacobians(model, X):
    Js, f = Jacobians(model, X)
    Js_naive, f_naive = Jacobians_naive(model, X)
    assert Js.shape == Js_naive.shape
    assert torch.abs(Js-Js_naive).max() < 1e-6
    assert torch.allclose(model(X), f_naive)
    assert torch.allclose(f, f_naive)
