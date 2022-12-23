import torch
import numpy as np

DATASET = "FMNIST"
MODEL_NAME = "CNN"
SEED = 117

PATH = '/home/mjazbec/laplace/BNN-predictions/experiments/'
fname = PATH + 'models/' + '_'.join([DATASET, MODEL_NAME, str(SEED)]) + '_{delta:.1e}.pt'

deltas = np.logspace(-2.0, 3.0, 16)
# deltas = np.insert(deltas, 0, 0)  # add unregularized network

LA_TYPE = 'map'

best_acc, best_nll, best_ece = 0., 1., 1.
for delta in deltas:
    print(delta)
    params = torch.load(fname.format(delta=delta), map_location=torch.device('cpu'))
    # print(params.keys())
    # print(params[LA_TYPE])
    acc = params[LA_TYPE]['acc_te']
    nll = params[LA_TYPE]['nll_te']
    ece = params[LA_TYPE]['ece_te']
    print(f"acc: {acc}",
          f"nll: {nll}",
          f"ece: {ece}")
    if acc >= best_acc:
        best_acc = acc
    if nll <= best_nll:
        best_nll = nll
    if ece <= best_ece:
        best_ece = ece
print(best_acc, best_nll, best_ece)