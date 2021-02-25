import pytest
import torch
import numpy as np
from scipy.linalg import block_diag

from backpack import backpack, extend
from backpack.extensions import KFLR

from preds.kron import Kron, symeig
from preds.utils import kronecker_product, get_sym_psd, kron_ggn


class MockModel:
    def __init__(self, seed=7):
        torch.manual_seed(seed)
        self.seed = seed
        self.bias = torch.randn(50)
        self.W = torch.randn(10, 25)
        self.conv = torch.randn(5, 3, 5, 5)
        self.n_params = 50 + 10*25 + 5*3*5*5

    def parameters(self):
        yield self.W
        yield self.bias
        yield self.conv

    def get_factors(self):
        torch.manual_seed(self.seed)
        B = get_sym_psd(50)
        W1, W2 = get_sym_psd(10), get_sym_psd(25)
        C1, C2 = get_sym_psd(5), get_sym_psd(3*5*5)
        return [[W1, W2], [B], [C1, C2]]


class SmallMockModel:
    def __init__(self, seed=10):
        torch.manual_seed(seed)
        self.seed = seed
        self.bias = torch.randn(2)
        self.W = torch.randn(2, 3)
        self.conv = torch.randn(2, 1, 2, 2)
        self.n_params = 2 + 2*3 + 2**3
        self.B = get_sym_psd(2)
        self.W1, self.W2 = get_sym_psd(2), get_sym_psd(3)
        self.C1, self.C2 = get_sym_psd(2), get_sym_psd(1*2*2)

    def parameters(self):
        yield self.W
        yield self.bias
        yield self.conv

    def get_factors(self):
        return [[self.W1, self.W2], [self.B], [self.C1, self.C2]]

    def get_prec(self):
        Prec = [kronecker_product(self.W1, self.W2).numpy(), self.B.numpy(),
                kronecker_product(self.C1, self.C2).numpy()]
        return torch.from_numpy(block_diag(*Prec))


@pytest.fixture
def model():
    from preds.models import CIFAR10Net
    torch.manual_seed(71)
    model = CIFAR10Net(in_channels=2, n_out=3)
    model = extend(model)
    return model


@pytest.fixture
def lossfunc():
    xent = torch.nn.CrossEntropyLoss(reduction='sum')
    return extend(xent)


@pytest.fixture
def X():
    torch.manual_seed(15)
    batch_size = 100
    channels = 2
    pixels = 28
    X = torch.randn(batch_size, channels, pixels, pixels)
    return X


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def small_mock_model():
    return SmallMockModel()


@pytest.fixture
def y():
    torch.manual_seed(15)
    batch_size = 100
    classes = 3
    y = torch.softmax(torch.randn(batch_size, classes), dim=1).argmax(dim=1)
    return y


def test_init_factors(model, lossfunc, X, y):
    kron = Kron(model, 0.03)
    model.zero_grad()
    loss = lossfunc(model(X), y)

    with backpack(KFLR()):
        loss.backward()

    for p, q in zip(model.parameters(), kron.qhs):
        for m, w in zip(p.kflr, q):
            assert m.shape == w.shape


def test_kron_aggregation(model, lossfunc, X, y):
    # kron should sum up the factors
    # we test if 2 * x = x + x
    kron = Kron(model, 0.3)
    for i in range(2):
        model.zero_grad()
        loss = lossfunc(model(X), y)

        with backpack(KFLR()):
            loss.backward()

        kron.update(kron_ggn(model, stochastic=False))

    for p, q in zip(model.parameters(), kron.qhs):
        for m, w in zip(p.kflr, q):
            assert torch.equal(2 * m, w)


def test_kron_batching(model, lossfunc, X, y):
    kron1 = Kron(model, 0.3)
    model.zero_grad()
    h = int(len(X) / 2)

    # update twice:
    loss = lossfunc(model(X[:h]), y[:h])
    with backpack(KFLR()):
        loss.backward()
    kron1.update([p.kflr for p in model.parameters()],
                 factor=1., batch_factor=1/2)
    loss = lossfunc(model(X[h:]), y[h:])
    with backpack(KFLR()):
        loss.backward()
    kron1.update([p.kflr for p in model.parameters()],
                 factor=1., batch_factor=1/2)

    kron2 = Kron(model, 0.3)
    model.zero_grad()
    # update just once full batch
    loss = lossfunc(model(X), y)
    with backpack(KFLR()):
        loss.backward()
    kron2.update([p.kflr for p in model.parameters()],
                 factor=1., batch_factor=1)

    if len(kron1.qhs) != len(kron2.qhs):
        assert False
    for QH, QHother in zip(kron1.qhs, kron2.qhs):
        if len(QH) == 1 and len(QHother) == 1:
            assert torch.abs(QH[0] - QHother[0]).max() < 1e-3
        elif len(QH) == 2 and len(QHother) == 2:
            assert torch.abs(QH[0] - QHother[0]).max() < 1e-3
            assert torch.abs(QH[1] - QHother[1]).max() < 1e-3


def test_kron_decomposition(model, lossfunc, X, y):
    kron = Kron(model, 0.7)
    model.zero_grad()
    loss = lossfunc(model(X), y)

    with backpack(KFLR()):
        loss.backward()

    kron.update([p.kflr for p in model.parameters()])
    kron.decompose()

    for p, ws, ls in zip(kron.qhs, kron.Ws, kron.Lams):
        for Q, W, l in zip(p, ws, ls):
            recon = W @ torch.diag(l) @ W.T
            assert torch.max(torch.abs(Q - recon)) < 1e-1


def test_kron_decomposition_mock(mock_model):
    kron = Kron(mock_model, 0.8)
    kron.update(mock_model.get_factors())
    kron.decompose()

    for p, ws, ls in zip(kron.qhs, kron.Ws, kron.Lams):
        for Q, W, l in zip(p, ws, ls):
            recon = W @ torch.diag(l) @ W.T
            assert torch.max(torch.abs(Q - recon)) < 1e-1


def test_kron_decomposition_shape(mock_model):
    kron = Kron(mock_model, 0.8)
    kron.update(mock_model.get_factors())
    kron.decompose()

    for l, w in zip(kron.Ws, kron.Lams):
        assert len(l) == len(w)
        for ll, ww in zip(l, w):
            assert len(ll) == len(ww)


def test_sampling_shape(mock_model):
    delta = 1.7
    kron = Kron(mock_model, delta)
    kron.update(mock_model.get_factors())
    kron.decompose()

    samples = kron.sample(77)
    S, P = samples.size()
    assert S == 77
    assert P == mock_model.n_params


def test_sampling_cov(small_mock_model):
    mock = small_mock_model
    delta = 0.56
    kron = Kron(mock, delta)
    kron.update(mock.get_factors())
    kron.decompose()
    Cov = np.cov(kron.sample(1000000).cpu().numpy().T)

    Sigma = torch.inverse(mock.get_prec() + torch.eye(mock.n_params) * delta).numpy()
    assert np.max(np.abs(Cov - Sigma)) < 1e-3


def test_sampling_bias(small_mock_model):
    mock = small_mock_model
    delta = 0.56
    kron = Kron(mock, delta)
    kron.update(mock.get_factors())
    kron.decompose()
    Cov = np.cov(kron.sample(100000).cpu().numpy().T)

    Sigma = torch.inverse(mock.get_prec() + torch.eye(mock.n_params) * delta).numpy()
    assert np.max(np.abs(Cov[6:8, 6:8] - Sigma[6:8, 6:8])) < 1e-2


def test_sampling_linear(small_mock_model):
    mock = small_mock_model
    delta = 0.56
    kron = Kron(mock, delta)
    kron.update(mock.get_factors())
    kron.decompose()
    Cov = np.cov(kron.sample(100000).cpu().numpy().T)

    Sigma = torch.inverse(mock.get_prec() + torch.eye(mock.n_params) * delta).numpy()
    assert np.max(np.abs(Cov[:6, :6] - Sigma[:6, :6])) < 1e-3


def test_sampling_conv(small_mock_model):
    mock = small_mock_model
    delta = 0.56
    kron = Kron(mock, delta)
    kron.update(mock.get_factors())
    kron.decompose()
    Cov = np.cov(kron.sample(100000).cpu().numpy().T)

    Sigma = torch.inverse(mock.get_prec() + torch.eye(mock.n_params) * delta).numpy()
    assert np.max(np.abs(Cov[8:, 8:] - Sigma[8:, 8:])) < 1e-3


def test_jacobian_product(small_mock_model):
    mock = small_mock_model
    delta = 0.15
    batch_size = 5
    classes = 3
    kron = Kron(mock, delta)
    kron.update(mock.get_factors())
    kron.decompose()
    Sigma = torch.inverse(mock.get_prec() + torch.eye(mock.n_params) * delta)
    Js = torch.randn(batch_size, classes, mock.n_params)
    f_var = kron.functional_variance(Js)
    for fi_var, Ji in zip(f_var, Js):
        assert torch.max(torch.abs(fi_var - Ji @ Sigma @ Ji.T)) < 1e-5


def test_symeig():
    torch.manual_seed(100)
    X = torch.randn(1000, 5000)
    M = X @ X.T / 5000
    Ltr, Wtr = torch.symeig(M, eigenvectors=True)
    L, W = symeig(M)
    assert torch.abs(L - Ltr).max() / Ltr.max() < 1e-5
    rectr = Wtr @ torch.diag(Ltr) @ Wtr.T
    rec = W @ torch.diag(L) @ W.T
    assert torch.allclose(rectr, rec)
    assert torch.allclose(rec, M, rtol=1e-5, atol=1e-5)
