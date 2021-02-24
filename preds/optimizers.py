import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam

from preds.gradients import Jacobians


def GGN(model, likelihood, data, target=None, ret_f=False):
    Js, f = Jacobians(model, data)
    if target is not None:
        rs = likelihood.residual(target, f)
    Hess = likelihood.Hessian(f)
    m, p = Js.shape[:2]
    if len(Js.shape) == 2:
        k = 1
        Hess = Hess.reshape(m, k, k)
        if target is not None:
            rs = rs.reshape(m, k)
        Js = Js.reshape(m, p, k)
    if target is not None:
        if ret_f:
            return Js, Hess, rs, f
        return Js, Hess, rs
    else:
        return Js, Hess, f


def expand_prior_mu(prior_mu, P, device):
    if type(prior_mu) is float:
        return torch.ones(P, device=device) * prior_mu
    elif type(prior_mu) is torch.Tensor:
        return prior_mu
    else:
        raise ValueError('Invalid shape for prior mean')


def expand_prior_prec(prior_prec, P, device):
    if type(prior_prec) is float or prior_prec.ndim == 0:
        prec_diag = torch.ones(P, device=device) * prior_prec
        return torch.diag(prec_diag), torch.diag(1 / prec_diag)
    elif prior_prec.ndim == 1:
        return torch.diag(prior_prec), torch.diag(1 / prior_prec)
    elif prior_prec.ndim == 2:
        return prior_prec, torch.inverse(prior_prec)
    else:
        raise ValueError('Invalid shape for prior precision')


class LaplaceGGN(Adam):

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), prior_prec=1.0, prior_mu=0.0,
                 eps=1e-8, amsgrad=False, **kwargs):
        if 'beta1' in kwargs and 'beta2' in kwargs:
            betas = (kwargs['beta1'], kwargs['beta2'])
        # prior precision penalty is added in step so there should be no wd here
        weight_decay = 0
        super(LaplaceGGN, self).__init__(model.parameters(), lr, betas, eps,
                                         weight_decay, amsgrad)
        p = parameters_to_vector(model.parameters())
        P = len(p)
        device = p.device
        self.defaults['device'] = device
        P_0, S_0 = expand_prior_prec(prior_prec, P, device)
        self.state['prior_prec'] = P_0
        self.state['Sigma_0'] = S_0
        self.state['prior_mu'] = expand_prior_mu(prior_mu, P, device)
        self.state['mu'] = None
        self.state['precision'] = None
        self.state['Sigma_chol'] = None

    def step(self, closure):
        # compute gradients on network using our standard closures
        log_lik = closure()
        params = parameters_to_vector(self.param_groups[0]['params'])
        prior_prec = self.state['prior_prec']
        weight_loss = 0.5 * params @ prior_prec @ params
        loss = - log_lik + weight_loss
        loss.backward()
        super(LaplaceGGN, self).step()
        return loss.item()

    def post_process(self, model, likelihood, train_loader):
        device = self.defaults['device']
        parameters = self.param_groups[0]['params']
        theta_star = parameters_to_vector(parameters).to(device)
        prior_prec = self.state['prior_prec']
        prior_mu = self.state['prior_mu']
        P = len(theta_star)
        JLJ = torch.zeros(P, P, device=device)
        G = torch.zeros(P, device=device)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            Js, Hess, rs = GGN(model, likelihood, data, target)
            JLJ += torch.einsum('mpk,mkl,mql->pq', Js, Hess, Js)
            G += torch.einsum('mpk,mk->p', Js, rs)
        # compute posterior covariance and precision
        self.state['precision'] = JLJ + prior_prec
        Chol = torch.cholesky(self.state['precision'])
        self.state['Sigma'] = torch.cholesky_inverse(Chol, upper=False)
        self.state['Sigma_chol'] = torch.cholesky(self.state['Sigma'], upper=False)
        # compute posterior mean according to BLR/Exact Laplace
        b = G + JLJ @ theta_star + prior_prec @ prior_mu
        self.state['mu'] = torch.cholesky_solve(b.reshape(-1, 1), Chol, upper=False).flatten()
        return self


def get_diagonal_ggn(optimizer):
    diag_prec = torch.diag(optimizer.state['precision'])
    Sigma_diag = 1 / diag_prec
    Sigma_chol = torch.diag(torch.sqrt(Sigma_diag))
    Sigma = torch.diag(Sigma_diag)
    return Sigma, Sigma_chol
