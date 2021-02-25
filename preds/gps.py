import numpy as np
import torch.multiprocessing as mp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, lazify
from gpytorch.distributions import MultivariateNormal
from opt_einsum import contract
import warnings

from collections import defaultdict
from contextlib import contextmanager
import itertools
from copy import deepcopy

from preds.gradients import Jacobians_gp

from preds.likelihoods import BernoulliLh, CategoricalLh

log2pi = np.log(2 * np.pi)


class Parallel_replicate_jacs(nn.Module):
    def __init__(self, J):
        super(Parallel_replicate_jacs, self).__init__()
        self.J = nn.Parameter(J, requires_grad=False)


class Parallel_Jacobian(nn.Module):
    def __init__(self, model, model_dim, output=0, method='batched'):
        super(Parallel_Jacobian, self).__init__()
        self.model = model
        self.output = output
        self.model_dim = model_dim
        self.method = method

    def forward(self, X):
        if self.method == 'batched':
            Js, fs = Jacobians_gp(self.model, X, self.output)
            if fs.ndim == 1:
                fs = fs.view(-1, 1)
        else:
            Js = torch.zeros(X.shape[0], self.model_dim, device=X.device)
            fs = torch.zeros(X.shape[0], self.model.output_size, device=X.device)
            for i in range(len(X)):
                f = self.model(X[[i]])
                fs[i] = f[0].detach()
                f = f[0, self.output] if f.ndim == 2 else f[0]
                grad = torch.autograd.grad(f, (p for p in self.model.parameters() if p.requires_grad))
                Js[i] = torch.cat([g.flatten() for g in grad])

        return Js, fs


class Parallel_calc_kernel(nn.Module):
    def __init__(self, opt_slices_gen, non_opt_slices_gen, n_kernel_groups, with_prior_prec=True):
        super(Parallel_calc_kernel, self).__init__()
        self.opt_slices_gen = opt_slices_gen
        self.non_opt_slices_gen = non_opt_slices_gen
        self.n_kernel_groups = n_kernel_groups
        self.with_prior_prec = with_prior_prec

    def forward(self, J_row, J_col, calc_diag=False):
        device = J_row.device

        if not calc_diag:
            kernel_slice = torch.zeros(self.n_kernel_groups, J_row.shape[0], J_col.shape[0], device=device)
        else:
            kernel_slice = torch.zeros(self.n_kernel_groups, J_row.shape[0], device=device)

        einsum_opt = 'ij,kj->ik' if not calc_diag else 'ij,ij->i'
        for kernel_group, _, pslices in self.opt_slices_gen:
            kernel_slice_group = kernel_slice[kernel_group]
            for s in pslices:
                kernel_slice_group.add_(contract(einsum_opt, J_row[:, s], J_col[:, s]))

        einsum_opt = 'ij,kj->ik' if not calc_diag else 'ij,ij->i'
        if self.with_prior_prec:
            einsum_opt = ',' + einsum_opt
        for _, pvar, pslices in self.non_opt_slices_gen:
            kernel_slice_group = kernel_slice[-1]
            for s in pslices:
                c = torch.tensor(pvar, device=device) if self.with_prior_prec else None
                einsum_args = (c, J_row[:, s], J_col[:, s]) if self.with_prior_prec else (J_row[:, s], J_col[:, s])
                kernel_slice_group.add_(contract(einsum_opt, *einsum_args).float())

        return kernel_slice


class Parallel_calc_gp_function(nn.Module):
    def __init__(self, nn_params):
        super(Parallel_calc_gp_function, self).__init__()
        self.nn_params = nn.Parameter(nn_params, requires_grad=False)

    def forward(self, J):
        return contract('ij,j->i', J, self.nn_params)

class Parallel_calc_E_z(nn.Module):
    def forward(self, K_train_train, pi):
        A = K_train_train
        D_sqrt = pi.sqrt()
        A.mul_(D_sqrt.unsqueeze(1)).mul_(D_sqrt.unsqueeze(2))
        A.diagonal(dim1=-2, dim2=-1).add_(torch.ones_like(pi))
        torch.cholesky(A, out=A)
        z = A.diagonal(dim1=-2, dim2=-1).log().sum()
        torch.cholesky_solve(DiagLazyTensor(D_sqrt).evaluate(), A, out=A)
        A.mul_(D_sqrt.unsqueeze(2))
        E = A
        return E, z.view(1)

# TODO: can save memory if compute b is separated as it doesn't need M
class Parallel_calc_b_c_pred(nn.Module):
    def __init__(self, M):
        super(Parallel_calc_b_c_pred, self).__init__()
        self.M = nn.Parameter(M, requires_grad = False)

    def forward(self, E, K_train_test):
        b = contract('cnm,cni->cmi', E, K_train_test)
        del E
        del K_train_test
        c = torch.cholesky_solve(b, self.M)
        return b, c

class Parallel_cholesky(nn.Module):
    def __init__(self):
        super(Parallel_cholesky, self).__init__()

    def forward(self, K):
        for v in [0., 1e-3, 1e-2, 1e-1]:
            try:
                if v > 0:
                    K.diagonal(dim1=-2, dim2=-1).add_(v)
                L = torch.cholesky(K, out=K)
                if v > 0:
                    warnings.warn('K is singular - adding jitter {}'.format(v))
                break
            except RuntimeError as e:
                continue

        return L,

class Parallel_cholesky_solve(nn.Module):
    def __init__(self):
        super(Parallel_cholesky_solve, self).__init__()

    def forward(self, v, L):
        return torch.cholesky_solve(v.unsqueeze(-1), L).squeeze(-1),


class Parallel_einsum(nn.Module):
    def __init__(self, einsum_opt):
        super(Parallel_einsum, self).__init__()
        self.opt = einsum_opt

    def forward(self, *operands):
        return contract(self.opt, *operands),


class Parallel_calc_c_Newton(nn.Module):
    def forward(self, E, K_train_train, b):
        c = contract('cij,cjk,ck->ci', E, K_train_train, b)
        return c,


class Parallel_calc_a(nn.Module):
    def __init__(self, M, c_sum):
        super(Parallel_calc_a, self).__init__()
        self.M = nn.Parameter(M, requires_grad=False)
        self.c_sum = nn.Parameter(c_sum, requires_grad=False)

    def forward(self, E, b, c):
        C, N = E.shape[:2]
        a = b - c + (E.view(C * N, N) @ torch.cholesky_solve(self.c_sum, self.M)).view(C, N)
        return a,


class Parallel_calc_mu(nn.Module):
    def forward(self, K_train_train, K_train_test, pi, gp_f_train_map):
        jitters = [0, 1e-3, 1e-2, 1e-1, 1]
        for i, v in enumerate(jitters):
            try:
                K_train_train.diagonal(dim1=-2, dim2=-1).add_(v * torch.ones_like(pi))  # TODO: rewrite this
                L = torch.cholesky(K_train_train, out=K_train_train)
                if i > 0:
                    print('Added jitter of', v)
                break
            except RuntimeError as e:
                if i == len(jitters) - 1:
                    raise e
        alpha = torch.cholesky_solve(gp_f_train_map.squeeze(0).unsqueeze(-1), L.squeeze(0)).squeeze(-1)  # TODO  .squeeze(0) needed?
        if alpha.ndim == 1:
            alpha.unsqueeze_(0)
        mu = contract('cnm,cn->cm', K_train_test, alpha)
        return mu,


def apply_parallel(module, inputs, module_args=tuple(), device_ids=None, out_device_id=-1, batch_size=None, no_grad=True):
    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n)) if n > 0 else [-1]
    elif isinstance(device_ids, int):
        device_ids = [device_ids]
    if batch_size is None:
        batch_size = len(device_ids)
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    total_batch_dim = len(inputs[0])

    if isinstance(module, list):  # replicas already initialized
        replicas = module
    else:
        model = module(*module_args).to('cuda:0' if torch.cuda.is_available() and device_ids[0] != -1 else 'cpu')
        replicas = nn.parallel.replicate(model, device_ids)

    outputs = []

    for batch_slice in DataLoader(range(total_batch_dim), batch_size=batch_size):
        inputs_batch = [inp[batch_slice] for inp in inputs]
        scattered_inputs = nn.parallel.scatter(inputs_batch, device_ids)
        replicas = replicas[:len(scattered_inputs)]
        if no_grad:
            with torch.no_grad():
                scattered_outputs = nn.parallel.parallel_apply(replicas, scattered_inputs)
        else:
            scattered_outputs = nn.parallel.parallel_apply(replicas, scattered_inputs)

        outputs_batch = nn.parallel.gather(scattered_outputs, out_device_id)
        del scattered_inputs
        del scattered_outputs

        for i, output in enumerate(outputs_batch):
            if len(outputs) == i:
                outputs.append([])
            outputs[i].append(output)

    for i in range(len(outputs)):
        outputs[i] = torch.cat(outputs[i], dim=0) if len(outputs[i]) > 0 else outputs[i][0]

    return outputs if len(outputs) > 1 else outputs[0]


def GP_LBFGS(likelihood, K_train_train, y_train, gp_train_mean, f_init, lbfgs_kwargs,
                             batch_size_C=-1, device_ids=None, out_device_id=-1):
    """
    K_train_train: C x N x N matrix consisting of N x N train kernels for C functions
    y_train: N training targets (integers 0, ..., C-1)
    likelihood: likelihood to calculate the predicted probabilities
    gp_train_mean: C x N calculated as f_star - J_star @ theta_star
    f_train_gp: C x N calculated as J_star @ theta_star - starting point for optimization
    """

    C = K_train_train.shape[0]
    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n)) if n > 0 else [-1]
    device = 'cpu' if device_ids[0] == -1 else 'cuda:{}'.format(device_ids[0])
    out_device = 'cpu' if out_device_id == -1 else 'cuda:{}'.format(out_device_id)

    gp_train_mean = gp_train_mean.to(device)
    y_train = y_train.to(device)

    if batch_size_C == -1 or batch_size_C >= C:  # keep L on GPUs
        batch_size_C = C
        calc_cholesky_replicas = nn.parallel.replicate(Parallel_cholesky(), device_ids)
        cholesky_solve_replicas = nn.parallel.replicate(Parallel_cholesky_solve(), device_ids)
        scattered_K = nn.parallel.scatter(K_train_train, device_ids)
        L = [o[0] for o in nn.parallel.parallel_apply(calc_cholesky_replicas, scattered_K)]

    else:  # keep L in CPU (or out_device_id)
        L = apply_parallel(Parallel_cholesky, (K_train_train,), device_ids=device_ids,
                                   batch_size=batch_size_C, out_device_id=out_device_id)

    y = likelihood.expand_targets(y_train, K_train_train.shape[0]).t()
#     trajectory = [f_init.detach().clone().to(out_device)]
    f = f_init.detach().clone().to(device).requires_grad_()  # TODO put to some cuda device
    opt = torch.optim.LBFGS([f], **lbfgs_kwargs)
    loss_prev = torch.tensor(float('inf'))
#     opt = torch.optim.Adam([f], **lbfgs_kwargs)

    def lbfgs_closure():
        with torch.no_grad():
            pi = likelihood.inv_link(f)
            fm = f - gp_train_mean
            if batch_size_C == C:
                scattered_fm = nn.parallel.scatter(fm.unsqueeze(-1), device_ids)
                scattered_outputs = [o[0] for o in nn.parallel.parallel_apply(cholesky_solve_replicas, list(zip(scattered_fm, L)))]
                a = nn.parallel.gather(scattered_outputs, device, dim=0).squeeze(-1)
            else:
                a = apply_parallel(Parallel_cholesky_solve, (fm.unsqueeze(-1), L), device_ids=device_ids,
                                   batch_size=batch_size_C, out_device_id=device).squeeze(-1)
            loss = -calc_Psi(a, fm)
            if loss > loss_prev:
                opt.param_groups[0]['lr'] *= 0.9
                opt.zero_grad()
            else:
                f.grad = a - y + pi
                loss_prev.data = loss.data

        print(-loss.item())
        return loss

    def calc_Psi(a, fm):
        return -0.5 * (a * fm).sum() + (y * f).sum() - torch.logsumexp(f, dim=0).sum()

    opt.step(lbfgs_closure)
#     for _ in range(300):
#         loss = lbfgs_closure()
#         opt.step()

    return (f.detach() - gp_train_mean).to(out_device)


def GP_Newton_steps(likelihood, K_train_train, y_train, gp_train_mean, f_init, max_steps=25,
                             batch_size_C=-1, device_ids=None, out_device_id=-1):
    """
    K_train_train: C x N x N matrix consisting of N x N train kernels for C functions
    y_train: N training targets (integers 0, ..., C-1)
    likelihood: likelihood to calculate the predicted probabilities
    gp_train_mean: C x N calculated as f_star - J_star @ theta_star
    f_train_gp: C x N calculated as J_star @ theta_star - starting point for optimization
    """

    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n)) if n > 0 else [-1]
    device = 'cpu' if device_ids[0] == -1 else 'cuda:{}'.format(device_ids[0])
    out_device = 'cpu' if out_device_id == -1 else 'cuda:{}'.format(out_device_id)

    gp_train_mean = gp_train_mean.to(out_device)
    y_train = y_train.to(out_device)

    if batch_size_C == -1:
        batch_size_C = K_train_train.shape[0]

    y = likelihood.expand_targets(y_train, K_train_train.shape[0]).t()

    def calc_Psi(a, f):
        return -0.5 * (a * f).sum() + likelihood.log_likelihood(y_train.to(out_device), (f + gp_train_mean.to(out_device)).t()).sum()

    with torch.no_grad():
        f = (f_init.to(out_device) - gp_train_mean).contiguous()
        trajectory = [f]
        i = 0
        Psi = torch.tensor(float('-inf'))
        while True:
            i += 1
            pi = likelihood.inv_link((f + gp_train_mean).t()).t()
            E, z_new = apply_parallel(Parallel_calc_E_z, (K_train_train, pi),
                                  device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

            z_new = z_new.sum()
            M = torch.cholesky(E.sum(0).to(device)).to(out_device)
            b = pi * f - pi * (pi * f).sum(dim=0, keepdim=True) + y - pi

            c = apply_parallel(Parallel_calc_c_Newton, (E, K_train_train, b),
                                  device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

            a = apply_parallel(Parallel_calc_a, (E, b, c), (M, c.sum(dim=0).view(-1, 1)),
                               device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

            f_new = apply_parallel(Parallel_einsum, (K_train_train, a), ('cij,cj->ci',),
                               device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

            Psi_new = calc_Psi(a, f_new)

            if Psi_new > Psi:
                f = f_new
                z = z_new
                Psi = Psi_new
                trajectory.append(f)
                if i % 1 == 0:
                    print('Step {}: Obj: {:.2f}'.format(i, Psi_new.item()))
            else:
                print('Failed to improve the objective further. Returning last f')
                return f, Psi - z, trajectory

            if i == max_steps:
                print('Reached max iter!')
                return f, Psi - z, trajectory


def GP_predictive(likelihood, gp_train_mean, gp_test_mean, K_train_train, K_train_test, K_test_test, gp_train_f, diag_only=True,
                             E=None, M=None,
                             batch_size_C=-1, batch_size_M=-1,
                             device_ids=None, out_device_id=-1):
    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n)) if n > 0 else [-1]
    device = 'cpu' if device_ids[0] == -1 else 'cuda:{}'.format(device_ids[0])
    out_device = 'cpu' if out_device_id == -1 else 'cuda:{}'.format(out_device_id)

    if batch_size_C == -1:
        batch_size_C = K_train_train.shape[0]

    if batch_size_M == -1:
        batch_size_M = K_test_test.shape[1]

    if diag_only and K_test_test.ndim == 3:
        K_test_test = K_test_test.diagonal(dim1=-2, dim2=-1)

    if not diag_only and K_train_train.shape[0] > 1:
        raise NotImplementedError()

    with torch.no_grad():
        pi = likelihood.inv_link((gp_train_f + gp_train_mean).t()).t()

        mu = gp_test_mean.to(out_device).t() + apply_parallel(Parallel_calc_mu, (K_train_train, K_train_test, pi, gp_train_f),
                                               device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id).t()

        if isinstance(likelihood, CategoricalLh):

            if E is None:
                E, _ = apply_parallel(Parallel_calc_E_z, (K_train_train, pi),
                                      device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)
            if M is None:
                M = torch.cholesky(E.sum(0).to(device)).to(out_device)

            b, c = apply_parallel(Parallel_calc_b_c_pred, (E, K_train_test), (M,), batch_size=batch_size_C)
            Sigma = apply_parallel(Parallel_einsum, (b.permute(2, 0, 1), c.permute(2, 1, 0)), ('mcn,mnd->mcd',),
                                   device_ids=device_ids, batch_size=batch_size_M, out_device_id=out_device_id)
            diag_term = apply_parallel(Parallel_einsum, (K_train_test.permute(2, 0, 1), b.permute(2, 1, 0)), ('mcn,mnc->mc',),
                                       device_ids=device_ids, batch_size=batch_size_M, out_device_id=out_device_id)
            Sigma.diagonal(dim1=1, dim2=2).add_(K_test_test.to(out_device).t() - diag_term)

        elif isinstance(likelihood, BernoulliLh):
            W = pi * (1 - pi)
            Wsqrt = W.sqrt().to(device)
            K = K_train_train.squeeze(0).to(device)
            K.mul_(Wsqrt).mul_(Wsqrt.t())
            K.diagonal().add_(1.)
            L = torch.cholesky(K, out=K)
            Kt = K_train_test.squeeze(0).to(device)
            Kt.mul_(Wsqrt.view(-1, 1))
            v = torch.triangular_solve(Kt, L, upper=False).solution
            mu = mu.flatten()
            if diag_only:
                Sigma = DiagLazyTensor(K_test_test.flatten().to(device) - contract('nm,nm->m', v, v))
            else:
                Sigma = K_test_test.squeeze(0).to(device) - contract('nm,no->mo', v, v)

    return MultivariateNormal(mu.to(device), Sigma.to(device))



## Newton with line search based on GPML code

def brentmin(xlow, xupp, Nitmax, tol, f, alpha, dalpha):
    eps = 1e-8
    fa = f(xlow, alpha, dalpha)[0]
    fb = f(xupp, alpha, dalpha)[0]
    seps = eps**0.5
    c = 0.381966011
    a = xlow
    b = xupp
    v = a + c * (b - a)
    w = v
    xf = v
    d = 0.
    e = 0.
    x = xf
    fx, *vargs = f(x, alpha, dalpha)
    print('mid x', x, fx)
    funccount = 3

    fv = fx
    fw = fx
    xm = 0.5 * (a + b)
    tol1 = seps * abs(xf) + tol / 3.0
    tol2 = 2. * tol1

    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        gs = True
        if abs(e) > tol1:
            gs = False
            r = (xf - w) * (fx - fv).item()
            q = (xf - v) * (fx - fw).item()
            p = (xf - v) * q - (xf - w) * r
            q = 2. * (q - r)
            if q > 0.:
                p = -p
            q = abs(q)
            r = e
            e = d
            if (abs(p) < abs(0.5 * q * r)) and (p > q * (a - xf)) and (p < q * (b - xf)):
                d = p / q;
                x = xf + d;
                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = 1 if xm - xf >= 0 else -1
                    d = tol1 * si
            else:
                gs = True
        if gs:
            e = a - xf if xf >= xm else b - xf
            d = c * e

        si = 1 if d >= 0 else -1
        x = xf + si * max(abs(d), tol1)
        fu, *vargs = f(x, alpha, dalpha)
        funccount += 1
#         print(funccount, a, b, fu.item())

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            v = w; fv = fw
            w = xf; fw = fx;
            xf = x; fx = fu;
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fw or w == xf:
                v = w; fv = fw;
                w = x; fw = fu;
            elif fu <= fv or v == xf or v == w:
                v = x; fv = fu;

        xm = 0.5 * (a + b)
        tol1 = seps * abs(xf) + tol / 3.; tol2 = 2. * tol1

        if funccount >= Nitmax:
            break

        if max(max(fx, fu), fw) - min(min(fx, fu), fw) < tol:
            break

    if fa < fx and fa <= fb:
        xf = xlow; fx = fa;
    elif fb < fx:
        xf = xupp; fx = fb

    fmin = fx; xmin = xf;

    return xmin, fmin, funccount, vargs

def Psi(alpha, m, Ka_fun, likfun):
    f = Ka_fun(alpha)
    fm = f + m
    lp, pi = likfun(fm)
    psi = 0.5 * (alpha * f).sum() - lp
    return psi, f, fm, alpha, pi

def irls(alpha, m, Ka_fun, alpha_fun, likfun, max_iter=20, tol=1e-3):
    smin_line = 0
    smax_line = 2
    nmax_line = 30
    thr_line = 1e-5
    Psi_line = lambda s, alpha, dalpha: Psi(alpha + s * dalpha, m, Ka_fun, likfun)
    search_line = lambda alpha, dalpha: brentmin(smin_line, smax_line, nmax_line, thr_line, Psi_line, alpha, dalpha)

    Psi_new, f, _, _, pi = Psi(alpha, m, Ka_fun, likfun)
    Psi_old = torch.tensor(float('inf'))
    it = 0
    while Psi_old - Psi_new > tol and it < max_iter:
        Psi_old = Psi_new
        it += 1

        alpha_newton, _ = alpha_fun(pi, f)
        dalpha = alpha_newton - alpha
        _, Psi_new, fc, (f, fm, alpha, pi) = search_line(alpha, dalpha)
        print('old', 'new', Psi_old, Psi_new, fc)

    return alpha, Psi_new



def GP_Newton_line_search(likelihood, K_train_train, y_train, gp_train_mean, f_init, max_steps=25, tol=1e-3,
                             batch_size_C=-1, device_ids=None, out_device_id=-1):
    C, N = K_train_train.shape[:2]
    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n)) if n > 0 else [-1]
    device = 'cpu' if device_ids[0] == -1 else 'cuda:{}'.format(device_ids[0])
    out_device = 'cpu' if out_device_id == -1 else 'cuda:{}'.format(out_device_id)

    gp_train_mean = gp_train_mean.to(out_device)
    f_init = f_init.to(out_device)
    y_train = y_train.clone().to(out_device)
    y = likelihood.expand_targets(y_train, K_train_train.shape[0]).t()

    if batch_size_C == -1:
        batch_size_C = K_train_train.shape[0]

    def likfun(f):
        if isinstance(likelihood, BernoulliLh):
            f = f.flatten()
        pi = likelihood.inv_link(f.t()).t()
        lp = likelihood.log_likelihood(y_train, f.t()).sum()
        if isinstance(likelihood, BernoulliLh):
            pi = pi.view(1, -1)
        return lp, pi

    def calc_Ka(a):
        if isinstance(likelihood, CategoricalLh):
            return apply_parallel(Parallel_einsum, (K_train_train, a), ('cij,cj->ci',),
                                   device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)
        else:
            return (K_train_train.squeeze(0).to(device) @ a.to(device).t()).t().to(out_device)


    def calc_a(pi, f):
        if isinstance(likelihood, CategoricalLh):
            E, z = apply_parallel(Parallel_calc_E_z, (K_train_train, pi),
                                      device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

            z = z.sum()
            M = torch.cholesky(E.sum(0).to(device)).to(out_device)
            b = pi * f - pi * (pi * f).sum(dim=0, keepdim=True) + y - pi

            c = apply_parallel(Parallel_calc_c_Newton, (E, K_train_train, b),
                                  device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

            a = apply_parallel(Parallel_calc_a, (E, b, c), (M, c.sum(dim=0).view(-1, 1)),
                               device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)
        elif isinstance(likelihood, BernoulliLh):
            W = pi * (1 - pi)
            Wsqrt = W.sqrt().to(device)
            b = (W * f + y - pi).to(device)
            K = K_train_train.squeeze(0).to(device)
            WKb = contract('n,nm,m->n', Wsqrt.flatten(), K, b.flatten())
            K.mul_(Wsqrt).mul_(Wsqrt.t())
            K.diagonal().add_(1.)
            L = torch.cholesky(K, out=K)
            a = (b - Wsqrt * torch.cholesky_solve(WKb.unsqueeze(-1), L).t()).to(out_device)
            z = L.diagonal().log().sum()

        return a, z

    with torch.no_grad():
        lp, pi = likfun(f_init)

        f = f_init - gp_train_mean

        L = apply_parallel(Parallel_cholesky, (K_train_train,),
                              device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

        alpha = apply_parallel(Parallel_cholesky_solve, (f, L),
                              device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)

        if max_steps > 0:
            alpha, psi_new = irls(alpha, gp_train_mean, calc_Ka, calc_a, likfun, max_iter=max_steps, tol=tol)
            f = calc_Ka(alpha)
            lp, pi = likfun(f + gp_train_mean)
            if isinstance(likelihood, CategoricalLh):
                z = apply_parallel(Parallel_calc_E_z, (K_train_train, pi),
                                      device_ids=device_ids, batch_size=batch_size_C, out_device_id=out_device_id)[1]
                z = z.sum()
            elif isinstance(likelihood, BernoulliLh):
                _, z = calc_a(pi, f)
        else:
            _, z = calc_a(pi, f)

        mll = -0.5 * (f * alpha).sum() + lp - z

    return f, mll

####

def gp_quantities(model, X_train, X_test=None, outputs=-1, optimize_params=set(), only_pred_var=True,
                  max_ram_gb=10, jacobian_method='batched', save_jacobians=False, train_jacobians=None, collapse_groups=False,
                  with_prior_prec=True, batch_size_per_gpu=-1, device_ids=None, print_progress=False):

    assert isinstance(X_train, (torch.Tensor, torch.utils.data.Dataset)) or train_jacobians
    if X_test is not None:
        assert isinstance(X_test, (torch.Tensor, torch.utils.data.Dataset))
    assert np.any([hasattr(p, 'log_prior_var') for p in model.parameters() if p.requires_grad])
    assert jacobian_method in ('naive', 'batched')


    if device_ids is None:
        n = torch.cuda.device_count()
        device_ids = list(range(n)) if n > 0 else [-1]

    default_device = 'cuda:{}'.format(device_ids[0])

    if isinstance(outputs, int):
        if outputs >= 0 and outputs < model.output_size:
            outputs = [outputs]
        elif outputs == -1:
            outputs = list(range(model.output_size))
        else:
            raise ValueError('Incorrect outputs argument value: {}'.format(outputs))
    elif not all(0 <= o < model.output_size for o in outputs):
        raise ValueError('Incorrect outputs argument value: {}'.format(outputs))

    n_train = len(X_train) if train_jacobians is None else train_jacobians[0].shape[0]
    n_test = len(X_test) if X_test is not None else 0
    n = n_train + n_test
    n_outputs = len(outputs)

    log_prob = 0
    log_temp = model.log_temp if hasattr(model, 'log_temp') else torch.tensor(0.)

    nn_params = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).to(default_device)

    all_vars = set(p.log_prior_var for p in model.parameters() if p.requires_grad and hasattr(p, 'log_prior_var'))
    n_groups = len(all_vars)
    optimize_params = set(optimize_params) if not isinstance(optimize_params, torch.Tensor) else set([optimize_params])
    vars_to_optimize = optimize_params.intersection(all_vars)
    log_prior_vars = list(vars_to_optimize)

    n_kernel_groups = min(n_groups, len(vars_to_optimize) + 1)
    kernel_map = dict(zip(vars_to_optimize, range(len(vars_to_optimize))))

    total_kernel_size = (n_train**2 if train_jacobians is not None else 0) + n_train * n_test + (n_test if only_pred_var else n_test**2)
    total_kernel_size *= (n_kernel_groups * n_outputs)
    if max_ram_gb * 1024 ** 3 <= 4 * total_kernel_size:
        raise RuntimeError('More memory needed to store kernels!')

    opt_slices = defaultdict(list)
    non_opt_slices = defaultdict(list)
    i = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.log_prior_var in vars_to_optimize:
            if opt_slices[p.log_prior_var] and opt_slices[p.log_prior_var][-1].stop == i:
                opt_slices[p.log_prior_var][-1] = slice(opt_slices[p.log_prior_var][-1].start, i + p.nelement())
            else:
                opt_slices[p.log_prior_var].append(slice(i, i + p.nelement()))
        else:
            if non_opt_slices[p.log_prior_var] and non_opt_slices[p.log_prior_var][-1].stop == i:
                non_opt_slices[p.log_prior_var][-1] = slice(non_opt_slices[p.log_prior_var][-1].start, i + p.nelement())
            else:
                non_opt_slices[p.log_prior_var].append(slice(i, i + p.nelement()))
        i += p.nelement()

    jac_size = i
    opt_slices = [(kernel_map[var], var.exp().item(), slices) for var, slices in opt_slices.items()]
    non_opt_slices = [(-1, var.exp().item(), slices) for var, slices in non_opt_slices.items()]
    all_slices = opt_slices + non_opt_slices

    f_gp_train = torch.ones(n_outputs, n_train)
    f_star_train = torch.ones(n_outputs, n_train)

    K_train = torch.zeros(n_outputs, n_kernel_groups, n_train, n_train) if train_jacobians is None else None

    if X_test is not None:
        f_gp_test = torch.ones(n_outputs, n_test)
        f_star_test = torch.ones(n_outputs, n_test)
        K_train_test = torch.zeros(n_outputs, n_kernel_groups, n_train, n_test)
        K_test_shape = (n_outputs, n_kernel_groups, n_test) if only_pred_var else (n_outputs, n_kernel_groups, n_test, n_test)
        K_test = torch.zeros(K_test_shape)
    else:
        K_train_test = None
        K_test = None

    jac_max_memory = max_ram_gb * 1024 ** 3 - 4 * total_kernel_size
    max_jacs_num = int(jac_max_memory / (4 * jac_size))
    batch_size = min(max_jacs_num, n_train if train_jacobians is None else n_test)

    if batch_size < n_train and save_jacobians:
        raise RuntimeError('Not enough memory to save all Jacobians')

    J_row = torch.zeros(batch_size, jac_size, pin_memory=True)

    gpu_models = []
    for i in device_ids:
        device = 'cuda:{}'.format(i)
        gpu_models.append(Parallel_Jacobian(model, jac_size, 0, jacobian_method).to(device))

    calc_kernel_replicas = nn.parallel.replicate(Parallel_calc_kernel(opt_slices, non_opt_slices, n_kernel_groups, with_prior_prec), device_ids)

    for idx, output in enumerate(outputs):
        if print_progress:
            print('Output {}'.format(output))
        for gpu_model in gpu_models:
            gpu_model.output = output

        total_rows = 0
        total_rows_train = 0
        total_rows_test = 0
        row_idcs_loader = itertools.chain(
            DataLoader(range(n_train), batch_size=batch_size if train_jacobians is None else n_train, shuffle=False),
            DataLoader(range(n_test), batch_size=batch_size,
                       shuffle=False) if X_test is not None and not only_pred_var else ()
        )

        for row_idcs in row_idcs_loader:
            row_slice = slice(row_idcs[0].item(), row_idcs[-1].item() + 1)

            is_train_row = total_rows_train < n_train  # TODO rename to is_train_col / row
            total_cols = total_rows_train

            X = X_train if is_train_row else X_test

            if train_jacobians is None or not is_train_row:
                for idcs in DataLoader(range(len(row_idcs)), batch_size=batch_size_per_gpu * len(device_ids)): # TODO: change these data loaders to simple range()
                    row_start, row_stop = idcs[0].item(), idcs[-1].item() + 1
                    row_slice2 = slice(row_slice.start + row_start, row_slice.start + row_stop)
    #                 jac_batch = J_row[row_start: row_stop]
                    scattered_input = nn.parallel.scatter(X[row_slice2], device_ids)
                    J_row_batch, f_row_batch = zip(*nn.parallel.parallel_apply(gpu_models, scattered_input))
                    J_row[row_start: row_stop] = nn.parallel.gather(J_row_batch, -1, dim=0)

                    if is_train_row and row_slice.start == 0:
                        f_star_train[:, row_slice2] = nn.parallel.gather(f_row_batch, -1, dim=0).t()[outputs]
            else:
                J_row_saved = train_jacobians[output]


            total_train_next = total_rows_train + len(row_idcs)
            total_test_next = total_rows_test + (len(row_idcs) if not is_train_row else 0)
            col_idcs_loader = itertools.chain(
                # Iterate through train Jacobians that are stored in J_row and compute kernel
                DataLoader(range(total_rows_train, total_train_next), batch_size=batch_size_per_gpu * len(device_ids),
                           shuffle=False) if n_train - total_rows_train > 0 and train_jacobians is None else (),
                # Calculate train Jacobians that are not in J_row and compute kernel
                DataLoader(range(total_train_next, n_train), batch_size=batch_size_per_gpu * len(device_ids),
                           shuffle=False) if n_train - total_train_next > 0 and train_jacobians is None else (),
                # Iterate through test Jacobians that are stored in J_row and compute kernel
                DataLoader(range(total_rows_test, total_test_next), batch_size=batch_size_per_gpu * len(device_ids),
                           shuffle=False) if X_test is not None and not is_train_row else (),
                # Calculate test Jacobians that are not in J_row and compute kernel
                DataLoader(range(total_test_next, n_test), batch_size=batch_size_per_gpu * len(device_ids),
                           shuffle=False) if X_test is not None and total_test_next < n_test else ()
            )


            for col_idcs in col_idcs_loader:
                col_slice = slice(col_idcs[0].item(), col_idcs[-1].item() + 1)
                if print_progress:
                    print(row_slice, col_slice)
                is_train_col = total_cols < n_train and train_jacobians is None
                is_first = is_train_row and row_slice.start == 0
                X = X_train if is_train_col else X_test  # TODO implement dataset

                if is_train_col is is_train_row and col_slice.start < row_slice.stop:
                    row_slice2 = slice(col_slice.start - row_slice.start, col_slice.stop - row_slice.start)
                    J_col = nn.parallel.scatter(J_row[row_slice2], device_ids)
                else:
                    scattered_input = nn.parallel.scatter(X[col_slice], device_ids)
                    J_col, f_col = zip(*nn.parallel.parallel_apply(gpu_models, scattered_input))

                if not is_train_col and only_pred_var and is_first:
                    scattered_outputs = nn.parallel.parallel_apply(calc_kernel_replicas, list(zip(J_col, J_col)),
                                                                   kwargs_tup=({'calc_diag': True},) * len(device_ids))

                    K_test[idx, :, col_slice].copy_(nn.parallel.gather(scattered_outputs, -1, dim=scattered_outputs[0].ndim - 1))
                    del scattered_outputs

                if is_first:
                    calc_gp_f_replicas = nn.parallel.replicate(Parallel_calc_gp_function(nn_params), device_ids)
                    scattered_outputs = nn.parallel.parallel_apply(calc_gp_f_replicas, J_col)
                    f_gp_slice = f_gp_train[idx, col_slice] if is_train_col else f_gp_test[idx, col_slice]
                    f_gp_slice.copy_(nn.parallel.gather(scattered_outputs, -1, dim=0))

                    if not is_train_col or col_slice.start >= row_slice.stop:
                        f_star = f_star_train if is_train_col else f_star_test
                        f_star[:, col_slice] = nn.parallel.gather(f_col, -1, dim=0).t()[outputs]

                for idcs in DataLoader(range(len(row_idcs)), batch_size=batch_size_per_gpu):
                    jac_left_start, jac_left_stop = idcs[0].item(), idcs[-1].item() + 1
                    row_slice2 = slice(row_slice.start + jac_left_start, row_slice.start + jac_left_stop)
                    J_row_ = J_row_saved if train_jacobians is not None and is_train_row else J_row
                    jac_batch = J_row_[jac_left_start: jac_left_stop]

                    if is_train_col and is_train_row:
                        kernel_slice = K_train[idx, :, row_slice2, col_slice]
                    elif not is_train_col and not is_train_row:
                        kernel_slice = K_test[idx, :, row_slice2, col_slice]
                    else:
                        kernel_slice = K_train_test[idx, :, row_slice2, col_slice]

                    replicated_left_Jacs = [o.J for o in nn.parallel.replicate(Parallel_replicate_jacs(jac_batch.to(default_device)), device_ids)]
                    scattered_outputs = nn.parallel.parallel_apply(calc_kernel_replicas, list(zip(replicated_left_Jacs, J_col)))
                    kernel_slice.copy_(nn.parallel.gather(scattered_outputs, -1, dim=scattered_outputs[0].ndim-1))

                    if is_train_col is is_train_row:
                        K = K_train if is_train_col else K_test
                        K[idx, :, col_slice, row_slice2].copy_(kernel_slice.transpose(-1, -2))

                    del scattered_outputs
                    del replicated_left_Jacs

                del J_col


                total_cols += len(col_idcs)

            total_rows += len(row_idcs)
            total_rows_train = min(total_rows, n_train)
            total_rows_test = max(0, total_rows - n_train)

    if not save_jacobians:
        del J_row
    del gpu_models

    def _collapse_groups(K):
        # TODO: think on gradient computatation in combination with gpytorch model including multiple prior precisions
        if len(vars_to_optimize) > 0:
            if K.shape[1] > 1:
                kernel = K[:, -1].clone() if len(vars_to_optimize) < n_groups else torch.zeros_like(K[:, -1])
                for i in range(len(log_prior_vars)):
                    kernel += K[:, i] * log_prior_vars[i].exp().cpu().detach()
            else:
                kernel = K[:, 0].clone()
                if len(vars_to_optimize) > 0:
                    kernel *= log_prior_vars[0].exp().cpu().detach()
        else:
            kernel = K[:, 0].clone()

        return kernel


    quantities = {
        'K_train': _collapse_groups(K_train) if collapse_groups else K_train,
        'f_gp_train': f_gp_train,
        'f_star_train': f_star_train,
        'prior_var': (list(all_vars)[0].exp().item() if len(all_vars) == 1 else None) if collapse_groups else None  # TODO ensure this is stored correctly
    }

    if X_test is not None:
        quantities['K_train_test'] = _collapse_groups(K_train_test) if collapse_groups else K_train_test
        quantities['K_test'] = _collapse_groups(K_test) if collapse_groups else K_test
        quantities['f_gp_test'] = f_gp_test
        quantities['f_star_test'] = f_star_test

    if save_jacobians:
        quantities['jacobians'] = J_row

    return quantities


def mc_preds(pred_dist, n_samples=100, batch_size=100):
    assert n_samples % min(n_samples, batch_size) == 0, 'Number of samples should be smaller than batch size or divisible by batch size'
    batch_size = min(n_samples, batch_size)
    n_batches = n_samples // batch_size
    preds_mean = torch.zeros(pred_dist.mean.shape, device=pred_dist.mean.device)
    # for simiplicity calculate only diagonal variance (simplification for multioutput)
    preds_var_a = torch.zeros(pred_dist.mean.shape, device=pred_dist.mean.device)
    preds_var_e = torch.zeros(pred_dist.mean.shape, device=pred_dist.mean.device)
    total = 0
    for i in range(n_batches):
        if preds_mean.ndim > 1:
            preds = pred_dist.sample(torch.Size([batch_size])).softmax(dim=-1)
        else:
            preds = pred_dist.sample(torch.Size([batch_size])).sigmoid()
        new_mean = preds.mean(dim=0)
        new_var = preds.var(dim=0)
        new_total = total + batch_size
        new_preds_mean = (total / new_total) * preds_mean + (batch_size / new_total) * new_mean
        preds_var_e = (total / new_total) * preds_var_e + (batch_size / new_total) * new_var + (total * batch_size / new_total**2) * (preds_mean - new_mean)**2
        preds_var_a += (1 / n_batches) * (preds - preds**2).mean(dim=0)
        preds_mean = new_preds_mean

    return preds_mean, preds_var_a, preds_var_e


def init_model_for_gp(model, likelihood, precision=1., optimize_precision=True, temperature=1., optimize_temperature=False, stochastic_bias=True, device='cuda'):
    model.likelihood =  likelihood
    log_temp = torch.tensor(np.log(temperature), requires_grad=optimize_temperature, device=device, dtype=torch.float32)
    model.log_temp = log_temp
    log_var = torch.tensor(-np.log(precision), requires_grad=optimize_precision, device=device, dtype=torch.float32)
    # set same precision for all parameters
    for pname, p in model.named_parameters():
        if p.requires_grad and (stochastic_bias or 'bias' not in pname):
            setattr(p, 'log_prior_var', log_var)
    opt_params = [p for p in [log_var, log_temp, *model.likelihood.parameters()] if p.requires_grad]
    model.hyperparameters = lambda: opt_params


import warnings

import torch

from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import _OneDimensionalLikelihood


class BernoulliLogitLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF \Phi(x)). Given the identity \Phi(-x) =
    1-\Phi(x), we can write the likelihood compactly as:
    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi(yf)
        \end{equation*}
    """

    def forward(self, function_samples, **kwargs):
        output_probs = function_samples.sigmoid()
        return base_distributions.Bernoulli(probs=output_probs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist, **kwargs):
        prob_lambda = lambda function_samples: function_samples.sigmoid()
        prob = self.quadrature(prob_lambda, function_dist)
        return base_distributions.Bernoulli(probs=prob)

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        if torch.any(observations.eq(-1)):
            # Remove after 1.0
            warnings.warn(
                "BernoulliLikelihood.expected_log_prob expects observations with labels in {0, 1}. "
                "Observations with labels in {-1, 1} are deprecated.",
                DeprecationWarning,
            )
        else:
            observations = observations.mul(2).sub(1)
        # Custom function here so we can use log_normal_cdf rather than Normal.cdf
        # This is going to be less prone to overflow errors
        log_prob_lambda = lambda function_samples: torch.nn.functional.logsigmoid(function_samples.mul(observations))
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob
