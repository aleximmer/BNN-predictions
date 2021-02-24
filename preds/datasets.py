import os
import pickle
import numpy as np
from sklearn.datasets import make_moons
import sklearn.model_selection as modsel
from sklearn.preprocessing import StandardScaler
import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from preds import DATA_DIR

CIFAR10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                         (0.24703223, 0.24348513, 0.26158784))
])

MNIST_transform = transforms.ToTensor()


class SnelsonGen(data.Dataset):

    def __init__(self, train=True, n_samples=150, s_noise=0.3,
                 random_seed=6, filter_left=True, filter_right=False,
                 root=DATA_DIR, double=False):
        self.train = train
        self.seed = random_seed if train else random_seed - 1
        self.n_samples = n_samples
        self.s_noise = s_noise
        self.filter_left = filter_left
        self.filter_right = filter_right
        with open(root + '/fsnelson.spy', 'rb') as file:
            self.fsnelson = pickle.load(file)
        X, y = self.generate()
        if double:
            self.data = torch.from_numpy(X).double()
            self.targets = torch.from_numpy(y).double()
        else:
            self.data = torch.from_numpy(X).float()
            self.targets = torch.from_numpy(y).float()

    def generate(self):
        samples = self.n_samples * 10
        np.random.seed(self.seed)
        x_min, x_max = self.x_bounds
        xs = (x_max - x_min) * np.random.rand(samples) + x_min
        fs = self.fsnelson(xs)
        ys = fs + np.random.randn(samples) * self.s_noise
        if self.filter_left:
            xfilter = (xs <= 1.5) | (xs >= 2.4)
            xs, ys = xs[xfilter], ys[xfilter]
        if self.filter_right:
            xfilter = (xs <= 4.4) | (xs >= 5.2)
            xs, ys = xs[xfilter], ys[xfilter]
        xs, ys = xs[:self.n_samples], ys[:self.n_samples]
        return xs.reshape(-1, 1), ys

    @property
    def x_bounds(self):
        x_min, x_max = (0.059167804, 5.9657729)
        return x_min, x_max

    @property
    def y_bounds(self):
        return min(self.targets).item(), max(self.targets).item()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.n_samples


class TwoMoons(data.Dataset):

    def __init__(self, train=True, random_seed=6, noise=0.3, n_samples=150,
                 double=False):
        self.train = train
        self.seed = random_seed if train else random_seed - 1
        X, y = make_moons(n_samples=n_samples, noise=noise,
                          random_state=self.seed)
        self.C = 2  # binary problem

        if double:
            self.data = torch.from_numpy(X).double()
            self.targets = torch.from_numpy(y).double()
        else:
            self.data = torch.from_numpy(X).float()
            self.targets = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


class CIFAR10(dset.CIFAR10):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=CIFAR10_transform, double=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.K = 10
        self.pixels = 32
        self.channels = 3
        if double:
            self.data = self.data.astype(np.double)


class SVHN(dset.SVHN):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=CIFAR10_transform):
        split = 'train' if train else 'test'
        super().__init__(root=root, split=split, download=download,
                         transform=transform)
        self.K = 10
        self.pixels = 32
        self.channels = 3


class MNIST(dset.MNIST):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=MNIST_transform, double=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.K = 10
        self.pixels = 28
        self.channels = 1
        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()


class FMNIST(dset.FashionMNIST):

    def __init__(self, root=DATA_DIR, train=True, download=True,
                 transform=MNIST_transform, double=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform)
        self.K = 10
        self.pixels = 28
        self.channels = 1
        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()


class UCIClassificationDatasets(data.Dataset):
    """UCIClassification data sets.
    The load wraps the data sets as used in https://github.com/cvillacampa/GPInputNoise
    and enables simple usage in torch.
    """

    def __init__(self, data_set, split_train_size=0.7, random_seed=6, shuffle=True, stratify=None,
                 root=DATA_DIR, train=True, valid=False, scaling=True, double=False):

        assert isinstance(random_seed, int), 'Please provide an integer random seed'
        error_msg = 'invalid UCI classification dataset'
        assert data_set in ['australian', 'breast_cancer', 'glass', 'ionosphere', 'vehicle',
                            'waveform', 'satellite', 'digits', 'banana'], error_msg

        assert isinstance(split_train_size, float), 'split_train_size can only be float'

        assert 0.0 <= split_train_size <= 1.0, 'split_train_size does not lie between 0 and 1'

        self.root = root
        self.train = train
        self.valid = valid
        if data_set in ['australian', 'breast_cancer']:
            if data_set == 'australian':
                aus = 'australian_presplit'
                x_train_file = os.path.join(self.root, aus, 'australian_scale_X_tr.csv')
                x_test_file = os.path.join(self.root, aus, 'australian_scale_X_te.csv')
                y_train_file = os.path.join(self.root, aus, 'australian_scale_y_tr.csv')
                y_test_file = os.path.join(self.root, aus, 'australian_scale_y_te.csv')
            else:
                bca = 'breast_cancer_scale_presplit'
                x_train_file = os.path.join(self.root, bca, "breast_cancer_scale_X_tr.csv")
                x_test_file = os.path.join(self.root, bca, "breast_cancer_scale_X_te.csv")
                y_train_file = os.path.join(self.root, bca, "breast_cancer_scale_y_tr.csv")
                y_test_file = os.path.join(self.root, bca, "breast_cancer_scale_y_te.csv")

            x_train, x_test = np.loadtxt(x_train_file), np.loadtxt(x_test_file)
            y_train, y_test = np.loadtxt(y_train_file), np.loadtxt(y_test_file)

        elif data_set == 'ionosphere':
            # hacky setting x_train, x_test
            filen = os.path.join(self.root, data_set, 'ionosphere.data')
            Xy = np.loadtxt(filen, delimiter=',')
            x_train, x_test = Xy[:50, :-1], Xy[50:, :-1]
            y_train, y_test = Xy[:50, -1], Xy[50:, -1]

        elif data_set == 'banana':
            # hacky setting x_train, x_test
            filen = os.path.join(self.root, data_set, 'banana.csv')
            Xy = np.loadtxt(filen, delimiter=',')
            x_train, y_train = Xy[:, :-1], Xy[:, -1]
            x_test, y_test = Xy[:0, :-1], Xy[:0, -1]
            y_train, y_test = y_train - 1, y_test - 1

        elif data_set == 'digits':
            # hacky setting x_train, x_test
            from sklearn.datasets import load_digits
            X, y = load_digits(return_X_y=True)
            x_train, x_test = X[:50], X[50:]
            y_train, y_test = y[:50], y[50:]

        else:
            # hacky setting x_train, x_test
            x_file = os.path.join(self.root, data_set, 'X.txt')
            y_file = os.path.join(self.root, data_set, 'Y.txt')
            X = np.loadtxt(x_file)
            y = np.loadtxt(y_file)
            x_train, x_test = X[:50], X[50:]
            y_train, y_test = y[:50], y[50:]

        x_full, y_full = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
        strat = y_full if stratify else None
        x_train, x_test, y_train, y_test = modsel.train_test_split(
            x_full, y_full, train_size=split_train_size, random_state=random_seed,
            shuffle=shuffle, stratify=strat)
        strat = y_test if stratify else None
        x_test, x_valid, y_test, y_valid = modsel.train_test_split(
            x_test, y_test, train_size=0.5, random_state=random_seed,
            shuffle=shuffle, stratify=strat)
        assert (len(y_test) + len(y_valid) + len(y_train)) == len(y_full)
        assert (len(x_test) + len(x_valid) + len(x_train)) == len(x_full)

        if scaling:
            self.scl = StandardScaler(copy=False)
            self.scl.fit_transform(x_train)
            self.scl.transform(x_test)
            self.scl.transform(x_valid)

        # impossible setting: if train is false, valid needs to be false too
        assert not (self.train and self.valid)
        if self.train:
            self.data, self.targets = torch.from_numpy(x_train), torch.from_numpy(y_train)
        else:
            if self.valid:
                self.data, self.targets = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
            else:
                self.data, self.targets = torch.from_numpy(x_test), torch.from_numpy(y_test)

        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()
        else:
            self.data = self.data.float()
            self.targets = self.targets.float()

        self.C = len(self.targets.unique())
        # for multiclass
        if self.C > 2:
            self.targets = self.targets.long()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]
