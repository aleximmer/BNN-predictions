import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.distributions import Normal
from backpack import backpack, extend
from backpack.extensions import DiagGGNExact, KFLR

from preds.optimizers import GGN
from preds.predictives import functional_sampling_predictive, nn_sampling_predictive
from preds.kron import Kron
from preds.gps import gp_quantities, GP_predictive, mc_preds


def diag_ggn(model):
    return torch.cat([p.diag_ggn_exact.data.flatten() for p in model.parameters()])


class Laplace:

    def __init__(self, model, prior_prec, likelihood):
        self.model = model
        self.prior_prec = prior_prec
        self.likelihood = likelihood
        self.mu = None
        self.Sigma_chol = None
        self.device = next(model.parameters()).device
        self.inferred = False
        self.P = len(parameters_to_vector(model.parameters()))

    def infer(self, train_loader, cov_type='full', dampen_kron=False):
        """compute the mean and covariance matrix
        cov_type either of ['full', 'diag', 'kron']
        """
        self.mu = parameters_to_vector(self.model.parameters()).detach()
        self.model.eval()  # e.g. turn off dropout

        if cov_type == 'full':
            precision = self.expand_prior_precision(cov_type)
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                Js, Hess, rs, f = GGN(self.model, self.likelihood, X, y, ret_f=True)
                precision += torch.einsum('mpk,mkl,mql->pq', Js, Hess, Js)
            Chol = torch.cholesky(precision)
            Sigma = torch.cholesky_inverse(Chol, upper=False)
            self.Sigma_chol = torch.cholesky(Sigma, upper=False)

        elif cov_type == 'diag':
            model = extend(self.model)
            nn_loss, hess_factor = self.likelihood.nn_loss()
            lossfunc = extend(nn_loss)
            precision = self.expand_prior_precision(cov_type)
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                model.zero_grad()
                f = model(X)
                loss = lossfunc(f, y)
                with backpack(DiagGGNExact()):
                    loss.backward()
                precision += hess_factor * diag_ggn(model)
            self.Sigma_chol = torch.sqrt(1 / precision)

        elif cov_type == 'kron':
            model = extend(self.model)
            nn_loss, hess_factor = self.likelihood.nn_loss()
            lossfunc = extend(nn_loss)
            assert np.isscalar(self.prior_prec)
            precision = Kron(model, self.prior_prec, dampen=dampen_kron)
            N = sum([len(a[0]) for a in train_loader]) if type(train_loader) is list else len(train_loader.dataset)
            for X, y in train_loader:
                M = len(y)
                X, y = X.to(self.device), y.to(self.device)
                model.zero_grad()
                f = model(X)
                loss = lossfunc(f, y)
                with backpack(KFLR()):
                    loss.backward()
                # update precision
                precision.update([p.kflr for p in model.parameters()], factor=hess_factor, batch_factor=M/N)
            precision.decompose()
            self.Sigma_chol = precision

        self.inferred = True

    def predictive_samples_glm(self, X, n_samples=1000):
        if not self.inferred:
            raise ValueError('Need to infer first.')
        if type(self.Sigma_chol) is not Kron:
            Sigma = self.Sigma
        else:
            Sigma = self.Sigma_chol
        return functional_sampling_predictive(X, self.model, self.likelihood, self.mu,
                                              Sigma, n_samples)

    def predictive_samples_bnn(self, X, n_samples=1000):
        if not self.inferred:
            raise ValueError('Need to infer first.')
        if type(self.Sigma_chol) is not Kron:
            Sigma_chol = self.Sigma_chol
        else:
            Sigma_chol = self.Sigma_chol
        return nn_sampling_predictive(X, self.model, self.likelihood, self.mu,
                                      Sigma_chol, n_samples)

    @property
    def Sigma(self):
        if type(self.Sigma_chol) is Kron:
            return self.Sigma_chol
        if self.Sigma_chol.ndim == 1:
            return self.Sigma_chol.square()
        elif self.Sigma_chol.ndim == 2:
            return self.Sigma_chol @ self.Sigma_chol.T
        else:
            raise ValueError('yet to be implemented for kron')

    def expand_prior_precision(self, cov_type='full'):
        # expands prior precision into full or diagonal matrix
        assert cov_type in ['full', 'diag']
        if np.isscalar(self.prior_prec) or self.prior_prec.ndim == 0:
            prec_diag = torch.ones(self.P, device=self.device) * self.prior_prec
            if cov_type == 'full':
                return torch.diag(prec_diag)
            elif cov_type == 'diag':
                return prec_diag
        elif self.prior_prec.ndim == 1:
            if cov_type == 'diag':
                return self.prior_prec
            elif cov_type == 'full':
                return torch.diag(self.prior_prec)
        elif self.prior_prec.ndim == 2:
            if cov_type == 'diag':
                raise ValueError('Will not diagonalize non-diagonal prior.')
            elif cov_type == 'full':
                return self.prior_prec
        else:
            raise ValueError('Invalid shape for prior precision')


class FunctionaLaplace:

    def __init__(self, model, prior_prec, likelihood, cache_Js=True):
        assert np.isscalar(prior_prec)
        self.model = model
        self.prior_prec = prior_prec
        self.likelihood = likelihood
        self.device = next(model.parameters()).device  # bit hacky but ok
        self.Kwinvs = None
        self.cache_Js = cache_Js
        self.Js = None
        self.train_loader = None

        self.log_likelihood = None
        self.inferred = False
        self.P = len(parameters_to_vector(model.parameters()))

        # need to set the below attributes for the gp_quantities method to work correctly
        model.likelihood = likelihood
        log_var = -torch.log(torch.tensor(prior_prec))

        for p in model.parameters():
            setattr(p, 'log_prior_var', log_var)

    def infer(self, train_loadera, train_loaderb, max_ram_gb=10, batch_size_per_gpu=512,
              print_progress=False):
        self.train_loader = train_loadera
        X1 = torch.cat([batch[0] for batch in train_loadera], dim=0).pin_memory()
        X2 = torch.cat([batch[0] for batch in train_loaderb], dim=0).pin_memory()

        self.gp_quantities = gp_quantities(self.model, X1, X2, collapse_groups=True,
                                           with_prior_prec=False,
                                           batch_size_per_gpu=batch_size_per_gpu,
                                           print_progress=print_progress, max_ram_gb=max_ram_gb)

        self.inferred = True

    def predictive_samples(self, n_samples=1000, batch_size=1024, indep=False):
        self.model.eval()
        if not self.inferred or 'f_star_test' not in self.gp_quantities:
            raise ValueError('Need to infer first.')

        K_nn = self.gp_quantities['K_train'] / self.prior_prec
        K_nm = self.gp_quantities['K_train_test'] / self.prior_prec
        K_m = self.gp_quantities['K_test'] / self.prior_prec
        m_test = self.gp_quantities['f_star_test']
        m_train = self.gp_quantities['f_star_train']
        if indep:
            p = torch.softmax(m_train, dim=0)
            w_sqrt = torch.sqrt(p * (1 - p))
            # B = I + W^1/2 K W^1/2 (tested Implementation) GP 3.4.3
            B = w_sqrt.unsqueeze(2) * K_nn * w_sqrt.unsqueeze(1)
            B.diagonal(dim1=1, dim2=2).add_(1.)
            Binvs = torch.inverse(B)
            # (K+W^-1)^-1 = W^1/2 B^-1 W^1/2  GP 3.4.3 (3.28)
            Kinvs = w_sqrt.unsqueeze(2) * Binvs * w_sqrt.unsqueeze(1)
            # predictive is N(m, K_** - K*n (K+W^-1)^-1 Kn*)
            # with Gain = K*n (K+W^-1)^-1 Kn*
            Gain = torch.einsum('cnm,cnp,cpm->cm', K_nm, Kinvs, K_nm)
            f_var = K_m - Gain
            fdist = Normal(m_test.T, f_var.T.sqrt())
        else:
            gp_train_mean = self.gp_quantities['f_star_train'] - self.gp_quantities['f_gp_train']
            gp_test_mean = self.gp_quantities['f_star_test'] - self.gp_quantities['f_gp_test']
            fdist = GP_predictive(self.likelihood, gp_train_mean, gp_test_mean, K_nn,
                                  K_nm, K_m, self.gp_quantities['f_gp_train'], out_device_id=-1,
                                  diag_only=True, batch_size_C=1, batch_size_M=batch_size)
            fdist.loc = self.gp_quantities['f_star_test'].T.cuda()
        pred_mc_test, _, _ = mc_preds(fdist, n_samples=n_samples, batch_size=batch_size)
        return pred_mc_test
