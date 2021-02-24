import torch
from torch.optim import Adam

from preds.models import SiBayesianMLP
from preds.likelihoods import BernoulliLh, GaussianLh


def preds_bbb(model, X, n_samples, likelihood):
    gs = torch.stack([likelihood.inv_link(model(X)) for _ in range(n_samples)])
    if type(likelihood) is GaussianLh:
        f_mu = gs.mean(dim=0)
        f_var = gs.var(dim=0)
        return f_mu, f_var
    else:
        g_mu = gs.mean(dim=0)
        return g_mu


def run_bbb(ds_train, ds_test, ds_valid, prior_prec, device, likelihood, n_epochs, lr,
            n_samples_train=10, n_samples_pred=1000, n_layers=2, n_units=50, activation='tanh'):
    D = ds_train.data.shape[1]
    if type(likelihood) is BernoulliLh or type(likelihood) is GaussianLh:
        K = 1
    else:
        K = ds_train.C
    model = SiBayesianMLP(D, K, n_layers, n_units, prior_prec, activation).to(device)
    X_train, y_train = ds_train.data.to(device), ds_train.targets.to(device)

    optim = Adam(model.parameters(), lr=lr)
    losses = []
    for i in range(n_epochs):
        optim.zero_grad()
        neg_log_liks = [-likelihood.log_likelihood(y_train, model(X_train))
                        for _ in range(n_samples_train)]
        neg_log_liks = torch.stack(neg_log_liks, dim=0)
        exp_neg_loglik = neg_log_liks.mean()
        loss = exp_neg_loglik + model.kl_divergence()  # negative elbo
        loss.backward()
        optim.step()
        losses.append(loss.detach().item())

    X_test = ds_test.data.to(device)
    X_valid = ds_valid.data.to(device)
    return {'preds_train': preds_bbb(model, X_train, n_samples_pred, likelihood),
            'preds_test': preds_bbb(model, X_test, n_samples_pred, likelihood),
            'preds_valid': preds_bbb(model, X_valid, n_samples_pred, likelihood),
            'elbos': losses}
