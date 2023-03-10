import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init


class PeriodDiscriminator(nn.Module):
    def __init__(self, periods, in_channels, out_channels):
        super().__init__()
        self.periods = periods
        self.blocks = nn.ModuleList()
        cur_channels = 32
        for i in range(4):
            self.blocks.append(
                weight_norm(nn.Conv2d(in_channels, cur_channels, kernel_size=(5, 1),
                                      stride=(3, 1), padding=(2, 0))),
            )
            in_channels = cur_channels
            cur_channels = min(cur_channels * 4, out_channels)
        self.blocks.extend([
            weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1),
                                  padding=(2, 0)))
        ])
        self.out_layer = weight_norm(nn.Conv2d(out_channels, 1, kernel_size=(3, 1),
                                  padding=(1, 0)))
        self.relu = nn.LeakyReLU(0.1)
        self.flatten = nn.Flatten(start_dim=1)


    def forward(self, x):
        batch_size, num_channels, seq_len = x.shape
        feature_matrices = []

        if seq_len % self.periods != 0:
            pad = self.periods - (seq_len % self.periods)
            seq_len += pad
            x = F.pad(x, (0, pad), "reflect")

        x = x.view(batch_size, num_channels,
                   seq_len // self.periods, self.periods)
        for block in self.blocks:
            x = self.relu(block(x))
            feature_matrices.append(x)
        x = self.out_layer(x)
        feature_matrices.append(x)
        return self.flatten(x), feature_matrices


class MPD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            PeriodDiscriminator(2, in_channels, out_channels),
            PeriodDiscriminator(3, in_channels, out_channels),
            PeriodDiscriminator(5, in_channels, out_channels),
            PeriodDiscriminator(7, in_channels, out_channels),
            PeriodDiscriminator(11, in_channels, out_channels)
        ])
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d,)):
            init.normal_(layer.weight, 0, 0.01)
            init.constant_(layer.bias, 0)

    def forward(self, true_wav, gen_wav):
        res_true = [None for _ in range(len(self.blocks))]
        res_gen = [None for _ in range(len(self.blocks))]
        feature_matrices_true = [None for _ in range(len(self.blocks))]
        feature_matrices_gen = [None for _ in range(len(self.blocks))]

        for i, block in enumerate(self.blocks):
            res_true[i], feature_matrices_true[i] = block(true_wav)
            res_gen[i], feature_matrices_gen[i] = block(gen_wav)
        return res_true, res_gen, \
               feature_matrices_true, feature_matrices_gen
