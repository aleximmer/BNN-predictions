import math
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from deepobs.pytorch.testproblems.testproblems_utils import tfconv2d, \
    tfmaxpool2d, _truncated_normal_init


class BayesianLinear(nn.Module):

    def __init__(self, in_features, out_features, delta):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.delta = delta
        rho_factor = -7.
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
        self.weight_rho = nn.Parameter(torch.ones((out_features, in_features)) * rho_factor)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.1, 0.1))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * rho_factor)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias_mu, -bound, bound)

    def forward(self, input):
        gamma = input @ self.weight_mu.T
        weight_std = 1e-6 + F.softplus(self.weight_rho)
        delta_sqrt = torch.sqrt(input.pow(2) @ weight_std.T.pow(2))
        epsilon_delta = torch.normal(mean=torch.zeros_like(delta_sqrt), std=1.0)
        epsilon_b = torch.normal(mean=torch.zeros_like(self.bias_mu), std=1.0)
        act_out_weights = gamma + delta_sqrt * epsilon_delta
        bias_std = 1e-6 + F.softplus(self.bias_rho)
        act_out_bias = self.bias_mu + bias_std * epsilon_b
        return act_out_weights + act_out_bias.unsqueeze(0)

    def kl_divergence(self):
        kl_w = self._kl_gaussian(q_mu=self.weight_mu, q_sigma=1e-6 + F.softplus(self.weight_rho),
                                 p_mu=0.0, p_sigma=1/math.sqrt(self.delta))
        kl_b = self._kl_gaussian(q_mu=self.bias_mu, q_sigma=1e-6 + F.softplus(self.bias_rho),
                                 p_mu=0.0, p_sigma=1/math.sqrt(self.delta))
        return kl_w + kl_b

    def _kl_gaussian(self, q_mu, q_sigma, p_mu, p_sigma):
        var_ratio = (q_sigma / p_sigma).pow(2)
        t1 = ((q_mu - p_mu) / p_sigma).pow(2)
        return 0.5 * torch.sum((var_ratio + t1 - 1. - var_ratio.log()))


class SiBayesianMLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, n_units, delta, activation='tanh'):
        super(SiBayesianMLP, self).__init__()
        assert n_layers >= 1
        self.input_size = input_size
        self.output_size = output_size
        self.delta = delta
        self.in_layer = BayesianLinear(input_size, n_units, delta)
        self.out_layer = BayesianLinear(n_units, output_size, delta)
        if n_layers > 1:  # more than one hidden layer
            hls = [BayesianLinear(n_units, n_units, delta) for _ in range(n_layers-1)]
            self.hidden_layers = nn.ModuleList(hls)
        else:
            self.hidden_layers = []

        # Set activation function
        if activation == "relu":
            self.act = torch.relu
        elif activation == "tanh":
            self.act = torch.tanh

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.act(self.in_layer(x))
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        y = self.out_layer(x)
        return y.flatten() if self.output_size == 1 else y

    def kl_divergence(self):
        kl = self.in_layer.kl_divergence()
        for h in self.hidden_layers:
            kl += h.kl_divergence()
        kl += self.out_layer.kl_divergence()
        return kl


class MLPS(nn.Sequential):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", flatten=False, bias=True):
        super(MLPS, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError('invalid activation')

        if flatten:
            self.add_module('flatten', nn.Flatten())

        if len(hidden_sizes) == 0:
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, self.output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=bias))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], self.output_size, bias=bias))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", **kwargs):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            self.act = torch.relu
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "selu":
            self.act = F.elu

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size
                                                in in_outs])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        z = self.output_layer(out)
        return z.flatten() if self.output_size == 1 else z


class SiMLP(MLP):
    """Simplified MLP with simpler searchable parameters (width x depth)
    instead of individual lists.
    """

    def __init__(self, input_size, output_size, n_layers, n_units, activation='tanh', **kwargs):
        hidden_sizes = [n_units for _ in range(n_layers)]
        super().__init__(input_size, hidden_sizes, output_size, activation, **kwargs)


"""Models from DeepOBS benchmark suite:
https://github.com/fsschneider/DeepOBS/blob/develop/deepobs/pytorch/testproblems/
testproblems_modules.py
"""


class CIFAR100Net(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    called `net_cifar100_allcnnc`
    """

    def __init__(self, in_channels=3, n_out=100):
        super(CIFAR100Net, self).__init__()
        assert n_out in (10, 100)
        self.output_size = n_out
        # only supported for these

        # self.add_module('dropout1', nn.Dropout(p=0.2))

        self.add_module('conv1', tfconv2d(
            in_channels=in_channels, out_channels=96, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv2', tfconv2d(
            in_channels=96, out_channels=96, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv3', tfconv2d(in_channels=96, out_channels=96, kernel_size=3,
                                          stride=(2, 2), tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())

        # self.add_module('dropout2', nn.Dropout(p=0.5))

        self.add_module('conv4', tfconv2d(
            in_channels=96, out_channels=192, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu4', nn.ReLU())
        self.add_module('conv5', tfconv2d(
            in_channels=192, out_channels=192, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu5', nn.ReLU())
        self.add_module('conv6', tfconv2d(in_channels=192, out_channels=192, kernel_size=3,
                                          stride=(2, 2), tf_padding_type='same'))
        self.add_module('relu6', nn.ReLU())

        # self.add_module('dropout3', nn.Dropout(p=0.5))

        self.add_module('conv7', tfconv2d(
            in_channels=192, out_channels=192, kernel_size=3))
        self.add_module('relu7', nn.ReLU())
        self.add_module('conv8', tfconv2d(
            in_channels=192, out_channels=192, kernel_size=1, tf_padding_type='same'))
        self.add_module('relu8', nn.ReLU())
        self.add_module('conv9', tfconv2d(
            in_channels=192, out_channels=n_out, kernel_size=1, tf_padding_type='same'))

        self.add_module('avg', nn.AvgPool2d(kernel_size=(6, 6)))
        self.add_module('flatten', nn.Flatten())

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.1)
                nn.init.xavier_normal_(module.weight)


class CIFAR10Net(nn.Sequential):
    """
    Deepobs network with optional last sigmoid activation (instead of relu)
    In Deepobs called `net_cifar10_3c3d`
    """

    def __init__(self, in_channels=3, n_out=10, use_tanh=False):
        super(CIFAR10Net, self).__init__()
        self.output_size = n_out
        activ = nn.Tanh if use_tanh else nn.ReLU

        self.add_module('conv1', tfconv2d(
            in_channels=in_channels, out_channels=64, kernel_size=5))
        self.add_module('relu1', nn.ReLU())
        self.add_module('maxpool1', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv2', tfconv2d(
            in_channels=64, out_channels=96, kernel_size=3))
        self.add_module('relu2', nn.ReLU())
        self.add_module('maxpool2', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('conv3', tfconv2d(
            in_channels=96, out_channels=128, kernel_size=3, tf_padding_type='same'))
        self.add_module('relu3', nn.ReLU())
        self.add_module('maxpool3', tfmaxpool2d(
            kernel_size=3, stride=2, tf_padding_type='same'))

        self.add_module('flatten', nn.Flatten())

        self.add_module('dense1', nn.Linear(
            in_features=3 * 3 * 128, out_features=512))
        self.add_module('relu4', activ())
        self.add_module('dense2', nn.Linear(in_features=512, out_features=256))
        self.add_module('relu5', activ())
        self.add_module('dense3', nn.Linear(in_features=256, out_features=n_out))

        # init the layers
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_normal_(module.weight)

            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.xavier_uniform_(module.weight)
