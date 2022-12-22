import torch
import torch.nn as nn
import torch.nn.init as init

from ast import literal_eval
from src.model.generator.MRF import MRF
from torch.nn.utils import weight_norm


class Generator(nn.Module):
    def __init__(self, d_input, d_inner, k_u, k_r, D_r):
        super().__init__()
        k_u, k_r, D_r = literal_eval(k_u), literal_eval(k_r), literal_eval(D_r)
        self.input_layer = weight_norm(nn.Conv1d(d_input, d_inner,
                                                 kernel_size=7, stride=1, padding=3))
        self.relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.upsampling_blocks = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        curr_channels = d_inner
        for i in range(len(k_u)):
            self.upsampling_blocks.append(
                weight_norm(nn.ConvTranspose1d(in_channels=curr_channels,
                                               out_channels=curr_channels // 2,
                                               kernel_size=k_u[i], stride=k_u[i] // 2,
                                               padding=(k_u[i] - k_u[i] // 2) // 2))
            )
            curr_channels //= 2
            self.mrfs.append(MRF(curr_channels, k_r, D_r))

        self.output_layer = weight_norm(nn.Conv1d(curr_channels, 1, kernel_size=7, padding=3))
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
            init.normal_(layer.weight, 0, 0.01)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(len(self.upsampling_blocks)):
            x = self.relu(x)
            x = self.upsampling_blocks[i](x)
            x = self.mrfs[i](x)
        x = self.output_layer(self.relu(x))
        return self.tanh(x)
