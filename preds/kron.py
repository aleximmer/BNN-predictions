import torch
import numpy as np
import logging


def diagonal_add_scalar(X, value):
    if not X.device == torch.device('cpu'):
        indices = torch.cuda.LongTensor([[i, i] for i in range(X.shape[0])])
    else:
        indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)


class Kron:
    def __init__(self, model, delta, dampen=False):
        self.qhs = self.init_factors(model)
        self.Ws = None
        self.Lams = None
        # either one delta per parameter group or one for all
        self.deltas = None
        self.change_delta(delta)
        self.dampen = dampen

    @staticmethod
    def init_factors(model):
        # initializes krons for a given model (tested)
        qhs = list()
        for p in model.parameters():
            shape = p.size()
            if p.ndim == 1:  # vector (typically bias)
                QH = torch.zeros(shape[0], shape[0], device=p.device)
                qhs.append([QH, ])
            elif p.ndim == 2:  # matrix (typically fully connected)
                Q = torch.zeros(shape[0], shape[0], device=p.device)
                H = torch.zeros(shape[1], shape[1], device=p.device)
                qhs.append([Q, H])
            else:
                m, n = shape[0], int(np.prod(shape[1:]))
                Q = torch.zeros(m, m, device=p.device)
                H = torch.zeros(n, n, device=p.device)
                qhs.append([Q, H])
        return qhs

    def change_delta(self, new_delta):
        # used to init or update delta
        if np.isscalar(new_delta):
            new_delta = torch.tensor(new_delta)
        if new_delta.ndim == 0 or (new_delta.ndim == 1 and len(new_delta) == 1):
            self.deltas = new_delta.repeat(len(self.qhs))
        elif new_delta.ndim == 1:
            if len(new_delta) != len(self.qhs):
                raise ValueError('Deltas do not match with parameter groups.')
            self.deltas = new_delta
        else:
            raise ValueError('Invalid dimensionality of delta.')

    def update(self, krons, factor=1., batch_factor=1.):
        # adds kronecker factors to the existing ones (tested)
        # batch-factor should be M / N
        for kron, QH in zip(krons, self.qhs):
            if len(kron) == 1:
                QH[0] += kron[0] * factor
            elif len(kron) == 2:
                # c(A \kron B) = cA \kron B = A \kron cB
                QH[0] += kron[0] * factor
                QH[1] += kron[1] * batch_factor
            else:
                raise ValueError('..')

    def decompose(self):
        # eigendecompose all kronecker factors
        Ws, Lams = list(), list()
        for qh in self.qhs:
            w, l = list(), list()
            for m in qh:
                L, W = symeig(m)
                l.append(L)  # only positive reals
                w.append(W)
            Ws.append(w)
            Lams.append(l)
        self.Ws = Ws
        self.Lams = Lams

    def sample(self, s):
        psamples = list()
        for l, w, delta in zip(self.Lams, self.Ws, self.deltas):
            if len(l) == 1:
                # standard matrix sqrt sampling
                W, l, p = w[0], l[0], len(l[0])
                ldelta = torch.sqrt(1 / (l + delta)).reshape(-1, 1)
                eps = torch.randn(p, s, device=W.device)
                psamples.append((W @ (ldelta * (W.T @ eps))).T)
            elif len(l) == 2:
                # using kronecker-factor properties
                W1, W2 = w
                l1, l2 = l
                if self.dampen:
                    # Q \kron H = (Q + sqrt(delta) I_q) \kron (H + sqrt(delta) I_h)
                    # simply raise the eigenvalues
                    l1d, l2d = l1 + torch.sqrt(delta), l2 + torch.sqrt(delta)
                    D = torch.sqrt(1 / torch.ger(l1d, l2d)).unsqueeze(0)
                else:
                    D = torch.sqrt(1 / (torch.ger(l1, l2) + delta)).unsqueeze(0)
                m, n = len(l1), len(l2)
                eps = torch.randn(s, m, n, device=W1.device)
                eps = (W1.T @ eps @ W2) * D
                eps = W1 @ eps @ W2.T
                psamples.append(eps.reshape(s, m*n))
        return torch.cat(psamples, dim=1)

    def functional_variance(self, Js, hess_factor=1.):
        # compute functional variance J Sigma J\transpose with J in k x p batched
        B, K, P = Js.size()
        Js = Js.reshape(B*K, P)  # output dim and sample are indifferent
        cur_p = 0
        SJ = list()
        for l, w, delta in zip(self.Lams, self.Ws, self.deltas):
            if len(l) == 1:
                W, l, p = w[0], l[0] * hess_factor, len(l[0])
                ldelta = (1 / (l + delta)).reshape(-1, 1)
                Js_p = Js[:, cur_p:cur_p+p].T
                SJ.append((W @ (ldelta * (W.T @ Js_p))).T)
                cur_p += p
            elif len(l) == 2:
                # using kronecker-factor properties
                W1, W2 = w
                l1, l2 = l[0] * hess_factor, l[1]
                p = len(l1) * len(l2)
                if self.dampen:
                    l1d, l2d = l1 + torch.sqrt(delta), l2 + torch.sqrt(delta)
                    D_inv = (1 / torch.ger(l1d, l2d)).unsqueeze(0)
                else:
                    D_inv = (1 / (torch.ger(l1, l2) + delta)).unsqueeze(0)
                m, n = len(l1), len(l2)
                Js_p = Js[:, cur_p:cur_p+p].reshape(B*K, m, n)
                Js_p = (W1.T @ Js_p @ W2) * D_inv
                Js_p = W1 @ Js_p @ W2.T
                SJ.append(Js_p.reshape(B*K, m*n))
                cur_p += p
        SJ = torch.cat(SJ, dim=1).reshape(B, K, P)
        f_var = torch.bmm(Js.reshape(B, K, P), SJ.transpose(1, 2))
        return f_var


def symeig(M):
    """Symetric eigendecomposition avoiding failure cases by
    adding and removing jitter to the diagonal
    returns eigenvalues (l) and eigenvectors (W)
    """
    # could make double to get more precise computation
    # M = M.double()
    # and then below return L.float(), W.float()
    try:
        L, W = torch.symeig(M, eigenvectors=True)
    except RuntimeError:  # did not converge
        logging.info('SYMEIG: adding jitter, did not converge.')
        # use W L W^T + I = W (L + I) W^T
        M = diagonal_add_scalar(M, value=1.)
        try:
            L, W = torch.symeig(M, eigenvectors=True)
            L -= 1.
        except RuntimeError:
            stats = f'diag: {M.diagonal()}, max: {M.abs().max()}, '
            stats = stats + f'min: {M.abs().min()}, mean: {M.abs().mean()}'
            logging.info(f'SYMEIG: adding jitter failed. Stats: {stats}')
            exit()
    # eigenvalues of symeig at least 0
    L = L.clamp(min=0.0)
    # underflow leads to nans instead of zeros
    L[torch.isnan(L)] = 0.0
    W[torch.isnan(W)] = 0.0
    return L, W
