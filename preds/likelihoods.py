from torch.distributions import Bernoulli, Normal, Categorical
import torch

EPS = 1e-7


def get_Lams_Vys(lh, Hess):
    # For Gaussian is identity, for class it is p - p^2
    Lams = Hess * lh.sigma_2_dispersion
    # For Gaussian this is s^2, for class it is only Lams (p - p^2)
    Vys = Lams * lh.sigma_2_dispersion
    return Lams, Vys


class Likelihood:

    @property
    def sigma_2_dispersion(self):
        return 1.0

    def log_likelihood(self, y, f):
        raise NotImplementedError

    def residual(self, y, f):
        return y - self.inv_link(f)

    def Hessian(self, f):
        raise NotImplementedError

    def inv_link(self, f):
        raise NotImplementedError


class GaussianLh(Likelihood):

    def __init__(self, sigma_noise=1):
        self.sigma_noise = sigma_noise

    @property
    def sigma_2_dispersion(self):
        return self.sigma_noise ** 2

    def log_likelihood(self, y, f):
        dist = Normal(f, self.sigma_noise)
        return torch.sum(dist.log_prob(y))

    def residual(self, y, f):
        return (y - f) / self.sigma_2_dispersion

    def Hessian(self, f):
        assert f.size(1) == 1
        return torch.ones_like(f).unsqueeze(-1) / self.sigma_2_dispersion

    def inv_link(self, f):
        return f

    def nn_loss(self):
        return torch.nn.MSELoss(reduction='sum'), 1 / (2 * self.sigma_2_dispersion)


class BernoulliLh(Likelihood):

    def log_likelihood(self, y, f):
        dist = Bernoulli(logits=f)
        return torch.sum(dist.log_prob(y))

    def Hessian(self, f):
        p = torch.clamp(self.inv_link(f), EPS, 1 - EPS)
        return p * (1 - p)

    def inv_link(self, f):
        return torch.sigmoid(f)

    def nn_loss(self):
        raise ValueError('No extendable nn loss for backpack in Bernoulli case')


class CategoricalLh(Likelihood):

    def log_likelihood(self, y, f):
        dist = Categorical(logits=f)
        return torch.sum(dist.log_prob(y))

    def residual(self, y, f):
        y_expand = torch.zeros_like(f)
        ixs = torch.arange(0, len(y)).long()
        y_expand[ixs, y.long()] = 1
        return y_expand - self.inv_link(f)

    def Hessian(self, f):
        p = torch.clamp(self.inv_link(f), EPS, 1 - EPS)
        H = torch.diag_embed(p) - torch.einsum('ij,ik->ijk', p, p)
        return H

    def inv_link(self, f):
        return torch.softmax(f, dim=-1)

    def nn_loss(self):
        return torch.nn.CrossEntropyLoss(reduction='sum'), 1

