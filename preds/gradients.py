import torch
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import BatchGrad
from backpack.context import CTX


def gradient(model):
    grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
    return grad.detach()


def cleanup(module):
    for child in module.children():
        cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)


def Jacobians(model, data):
    # Jacobians are batch x params x output
    model = extend(model)
    to_stack = []
    for i in range(model.output_size):
        model.zero_grad()
        out = model(data)
        with backpack(BatchGrad()):
            if model.output_size > 1:
                out[:, i].sum().backward()
            else:
                out.sum().backward()
            to_cat = []
            for param in model.parameters():
                to_cat.append(param.grad_batch.detach().reshape(data.shape[0], -1))
            Jk = torch.cat(to_cat, dim=1)
        to_stack.append(Jk)
        if i == 0:
            f = out.detach()
    # cleanup
    model.zero_grad()
    CTX.remove_hooks()
    cleanup(model)
    if model.output_size > 1:
        return torch.stack(to_stack, dim=2), f
    else:
        return Jk, f


def Jacobians_gp(model, data, outputs=-1):
    model = extend(model)
    to_stack = []
    outputs = range(model.output_size) if outputs == -1 else ([outputs] if isinstance(outputs, int) else outputs)
    for i in outputs:
        model.zero_grad()
        out = model(data)
        with backpack(BatchGrad()):
            if model.output_size > 1:
                out[:, i].sum().backward()
            else:
                out.sum().backward()
            to_cat = []
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                to_cat.append(param.grad_batch.detach().reshape(data.shape[0], -1))
            Jk = torch.cat(to_cat, dim=1)
        to_stack.append(Jk)
        if i == outputs[0]:
            f = out.detach()
    # cleanup
    model.zero_grad()
    CTX.remove_hooks()
    cleanup(model)
    if len(outputs) > 1:
        return torch.stack(to_stack, dim=2), f
    else:
        return Jk, f


def Jacobians_naive(model, data):
    model.zero_grad()
    f = model(data)
    Jacs = list()
    for i in range(f.shape[0]):
        if len(f.shape) > 1:
            jacs = list()
            for j in range(f.shape[1]):
                rg = (i != (f.shape[0] - 1) or j != (f.shape[1] - 1))
                f[i, j].backward(retain_graph=rg)
                Jij = gradient(model)
                jacs.append(Jij)
                model.zero_grad()
            jacs = torch.stack(jacs).t()
        else:
            rg = (i != (f.shape[0] - 1))
            f[i].backward(retain_graph=rg)
            jacs = gradient(model)
            model.zero_grad()
        Jacs.append(jacs)
    Jacs = torch.stack(Jacs)
    return Jacs.detach().squeeze(), f.detach()
