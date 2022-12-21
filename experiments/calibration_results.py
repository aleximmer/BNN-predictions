import torch

params = torch.load("models/FMNIST_CNN_117_1.0e-02.pt", map_location=torch.device('cpu'))
print(params['lap_kron'])