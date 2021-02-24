import torch
from torch.distributions import MultivariateNormal, Normal
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import logging

from preds.gradients import Jacobians
from preds.optimizers import GGN
from preds.likelihoods import get_Lams_Vys, GaussianLh
from preds.kron import Kron


def nn_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=100, no_link=False):
    theta_star = parameters_to_vector(model.parameters())
    if type(Sigma_chol) is Kron:
        covs = Sigma_chol.sample(mc_samples)
    elif len(Sigma_chol.shape) == 2:
        covs = (Sigma_chol @ torch.randn(len(mu), mc_samples, device=mu.device)).t()
    elif len(Sigma_chol.shape) == 1:
        covs = (Sigma_chol.reshape(-1, 1) * torch.randn(len(mu), mc_samples, device=mu.device)).t()
    samples = mu + covs
    predictions = list()
    link = (lambda x: x) if no_link else likelihood.inv_link
    for i in range(mc_samples):
        vector_to_parameters(samples[i], model.parameters())
        f = model(X)
        predictions.append(link(f).detach())
    vector_to_parameters(theta_star, model.parameters())
    return torch.stack(predictions)


def linear_sampling_predictive(X, model, likelihood, mu, Sigma_chol, mc_samples=100, no_link=False):
    theta_star = parameters_to_vector(model.parameters())
    Js, f = Jacobians(model, X)
    if len(Js.shape) > 2:
        Js = Js.transpose(1, 2)
    offset = f - Js @ theta_star
    if len(Sigma_chol.shape) == 2:
        covs = (Sigma_chol @ torch.randn(len(mu), mc_samples, device=mu.device)).t()
    elif len(Sigma_chol.shape) == 1:
        covs = (Sigma_chol.reshape(-1, 1) * torch.randn(len(mu), mc_samples, device=mu.device)).t()
    samples = mu + covs
    predictions = list()
    link = (lambda x: x) if no_link else likelihood.inv_link
    for i in range(mc_samples):
        f = offset + Js @ samples[i]
        predictions.append(link(f).detach())
    return torch.stack(predictions)


def functional_sampling_predictive(X, model, likelihood, mu, Sigma, mc_samples=1000, no_link=False):
    theta_star = parameters_to_vector(model.parameters())
    Js, f = Jacobians(model, X)
    # reshape to batch x output x params
    if len(Js.shape) > 2:
        Js = Js.transpose(1, 2)
    else:
        Js = Js.unsqueeze(1)  # add the output dimension
    f_mu = f + Js @ (mu - theta_star)
    if type(Sigma) is Kron:
        # NOTE: Sigma is in this case not really cov but prec-kron and internally inverted
        f_var = Sigma.functional_variance(Js)
    elif len(Sigma.shape) == 2:
        f_var = torch.einsum('nkp,pq,ncq->nkc', Js, Sigma, Js)
    elif len(Sigma.shape) == 1:
        f_var = torch.einsum('nkp,p,ncp->nkc', Js, Sigma, Js)
    if type(likelihood) is GaussianLh:
        return f_mu, f_var
    else:
        link = (lambda x: x) if no_link else likelihood.inv_link
        try:
            fs = MultivariateNormal(f_mu, f_var)
            return link(fs.sample((mc_samples,)))
        except RuntimeError:
            logging.warning('functional sampling covariance indefinite - use diagonal')
            fs = Normal(f_mu, f_var.diagonal(dim1=1, dim2=2).clamp(1e-5))
            return link(fs.sample((mc_samples,)))


def linear_regression_predictive(X, model, likelihood, mu, Sigma_chol):
    theta_star = parameters_to_vector(model.parameters())
    if len(Sigma_chol.shape) == 2:
        Sigma = Sigma_chol @ Sigma_chol.t()
    elif len(Sigma_chol.shape) == 1:
        Sigma = torch.diag(Sigma_chol ** 2)
    Js, Hess, f = GGN(model, likelihood, X)
    Lams, Vys = get_Lams_Vys(likelihood, Hess)
    delta = mu - theta_star
    # Lam Js = Jacobians of inv link g (m x p x k)
    Jgs = torch.bmm(Js, Lams)
    lin_pred = torch.einsum('mpk,p->mk', Jgs, delta).reshape(*f.shape)
    mu_star = (likelihood.inv_link(f) + lin_pred).detach()
    var_f = torch.bmm(Jgs.transpose(1, 2) @ Sigma, Jgs).squeeze().detach()
    var_noise = Vys.squeeze().detach()
    return mu_star, var_f, var_noise
