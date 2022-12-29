import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('results/FMNIST_CNN_117_GP.pkl', 'rb') as f:
    data = pickle.load(f)

    data.pop('delta')
    _data = []
    for k, v in data.items():
        k = k.split("-")
        if v['perf'] is not None:
            _data.append((k[1], k[2], v['perf']['acc_te'], v['perf']['ece_te'], v['perf']['nll_te']))
        else:
            _data.append((k[1], k[2], None, None, None))
    data = pd.DataFrame(_data, columns=["M", "delta", "acc", "ECE", "NLL"])


deltas = list(data["delta"].unique())
fig, ax = plt.subplots(len(deltas), 2, figsize=(10, 30))
for i, delta in enumerate(deltas):
    print(f"======================= GP results for delta={delta} =======================")
    data_delta = data[data["delta"] == delta]
    print(data_delta)
    ax[i, 0].plot(data_delta["M"].values.astype(int), data_delta["ECE"].values.astype(float))
    ax[i, 0].set_title(f"ECE delta={float(delta):.2f}")
    ax[i, 1].plot(data_delta["M"].values.astype(int), data_delta["NLL"].values.astype(float))
    ax[i, 1].set_title(f"NLL delta={float(delta):.2f}")
plt.tight_layout()
plt.show()