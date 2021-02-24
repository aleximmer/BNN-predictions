import torch
from preds.gradients import Jacobians
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Normal, kl_divergence
from opt_einsum import contract as einsum


def laplace_refine(model, X, y, likelihood, prior_prec, n_epochs=1000, lr=1e-3):
    """Laplace in LinNN (GLM) giving posterior mean, and Sigma_chol(d)"""
    J, f = Jacobians(model, X)
    if len(J.shape) == 3:
        J = J.transpose(1, 2)
    theta_star = parameters_to_vector(model.parameters()).detach()
    f_offset = f - (J @ theta_star)
    mu = theta_star.clone().requires_grad_(True)
    opt = Adam([mu], lr=lr)

    losses = list()
    for i in range(n_epochs):
        opt.zero_grad()
        ll_loss = - likelihood.log_likelihood(y, f_offset + J @ mu)
        weight_loss = 0.5 * mu @ mu * prior_prec
        loss = ll_loss + weight_loss
        loss.backward()
        losses.append(loss.detach().cpu().item())
        opt.step()

    Lams = likelihood.Hessian(f_offset + J @ mu.detach())
    if len(J.shape) == 3:
        Hessian = einsum('mkp,mkl,mlq->pq', J, Lams, J, backend='torch')
    else:
        Hessian = (J.T * Lams.reshape(1, -1)) @ J
    Precision = Hessian + prior_prec * torch.eye(len(Hessian), device=Hessian.device)
    Chol = torch.cholesky(Precision)
    Sigma = torch.cholesky_inverse(Chol, upper=False)
    Sigma_chol = torch.cholesky(Sigma, upper=False)
    Sigma_chold = torch.diag(torch.sqrt(1 / torch.diag(Precision)))
    return mu.detach(), Sigma_chol, Sigma_chold, losses


def vi_refine(model, opt, X, y, likelihood, n_epochs=250, lr=1e-2):
    """VI in LinNN (GLM) giving posterior mean, and Sigma_chol(d)"""
    beta = lr
    J, f = Jacobians(model, X)
    if len(J.shape) == 3:
        J = J.transpose(1, 2)
    theta_star = parameters_to_vector(model.parameters()).detach()
    f_offset = f - (J @ theta_star)
    mu = theta_star.clone().requires_grad_(False)
    P = len(mu)
    Sigma_chol = opt.state['Sigma_chol'].clone()
    prior_prec = opt.state['prior_prec']
    prec = opt.state['precision'].clone()
    sigma_chol_prior = torch.diag(torch.sqrt(1 / torch.diag(prior_prec)))
    p = MultivariateNormal(torch.zeros_like(mu), scale_tril=sigma_chol_prior)

    losses = list()
    for i in range(n_epochs):
        theta_s = mu + Sigma_chol @ torch.randn(P, device=mu.device)
        f_t = f_offset + J @ theta_s
        Lams = likelihood.Hessian(f_t)
        rs = likelihood.residual(y, f_t)
        # Hessian of E[log p] and gradient of ELBO
        if len(J.shape) == 3:
            H = einsum('mkp,mkl,mlq->pq', J, Lams, J, backend='torch')
            g = einsum('mkp,mk->p', J, rs, backend='torch') - prior_prec @ mu
        else:
            H = (J.T * Lams.reshape(1, -1)) @ J
            g = J.T @ rs - prior_prec @ mu
        # Update
        prec = (1 - beta) * prec + beta * (H + prior_prec)
        pchol = torch.cholesky(prec, upper=False)
        b = torch.cholesky_solve(g.reshape(-1, 1), pchol, upper=False).squeeze()
        mu = mu + beta * b
        Sigma_chol = torch.cholesky(torch.cholesky_inverse(pchol, upper=False), upper=False)
        q = MultivariateNormal(loc=mu, scale_tril=Sigma_chol)
        loss = -likelihood.log_likelihood(y, f_t) + kl_divergence(q, p)
        losses.append(loss.detach().cpu().item())

    return mu, Sigma_chol, losses


def vi_diag_refine(model, opt, X, y, likelihood, n_epochs=250, lr=1e-3):
    """diag VI in LinNN (GLM) giving posterior mean, and Sigma_chol(d)"""
    beta = lr
    J, f = Jacobians(model, X)
    if len(J.shape) == 3:
        J = J.transpose(1, 2)
    theta_star = parameters_to_vector(model.parameters()).detach()
    f_offset = f - (J @ theta_star)
    mu = theta_star.clone().requires_grad_(False)
    P = len(mu)
    prec = torch.diag(opt.state['precision'].clone())
    prior_prec = torch.diag(opt.state['prior_prec'])
    sigma_chol = torch.sqrt(1 / prec)
    m_prior = Normal(loc=torch.zeros_like(mu), scale=1/torch.sqrt(prior_prec))

    losses = list()
    for i in range(n_epochs):
        theta_s = mu + sigma_chol * torch.randn(P, device=mu.device)
        f_t = f_offset + J @ theta_s
        Lams = likelihood.Hessian(f_t)
        rs = likelihood.residual(y, f_t)
        # Hessian of E[log p] and gradient of ELBO
        if len(J.shape) == 3:
            H = torch.diag(einsum('mkp,mkl,mlq->pq', J, Lams, J, backend='torch'))
            g = einsum('mkp,mk->p', J, rs, backend='torch') - prior_prec * mu
        else:
            H = torch.diag((J.T * Lams.reshape(1, -1)) @ J)
            g = J.T @ rs - prior_prec * mu
        # Update
        prec = (1 - beta) * prec + beta * (H + prior_prec)
        mu = mu + beta * g / prec
        sigma_chol = torch.sqrt(1 / prec)
        m_post = Normal(loc=mu, scale=sigma_chol)
        loss = -likelihood.log_likelihood(y, f_t) + kl_divergence(m_post, m_prior).sum()
        losses.append(loss.detach().cpu().item())

    return mu, torch.diag(sigma_chol), losses

