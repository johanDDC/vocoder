import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.init as init


class ScaleDiscriminator(nn.Module):
    def __init__(self, normalization_method):
        super().__init__()
        self.blocks = nn.ModuleList([
            normalization_method(nn.Conv1d(1, 128, kernel_size=15, padding=7)),
            normalization_method(nn.Conv1d(128, 128, kernel_size=41, stride=2,
                                           groups=4, padding=20)),
            normalization_method(nn.Conv1d(128, 256, kernel_size=41, stride=2,
                                           groups=16, padding=20)),
            normalization_method(nn.Conv1d(256, 512, kernel_size=41, stride=4,
                                           groups=16, padding=20)),
            normalization_method(nn.Conv1d(512, 1024, kernel_size=41, stride=4,
                                           groups=16, padding=20)),
            normalization_method(nn.Conv1d(1024, 1024, kernel_size=41, groups=16, padding=20)),
            normalization_method(nn.Conv1d(1024, 1024, kernel_size=5, padding=2)),
        ])
        self.out_layer = normalization_method(nn.Conv1d(1024, 1, kernel_size=3, padding=1))
        self.relu = nn.LeakyReLU(0.1)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        feature_matrices = []
        for block in self.blocks:
            x = self.relu(block(x))
            feature_matrices.append(x)
        x = self.out_layer(x)
        feature_matrices.append(x)
        return self.flatten(x), feature_matrices


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            ScaleDiscriminator(spectral_norm),
            ScaleDiscriminator(weight_norm),
            ScaleDiscriminator(weight_norm),
        ])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv1d,)):
            init.normal_(layer.weight, 0, 0.01)
            init.constant_(layer.bias, 0)

    def forward(self, true_wav, gen_wav):
        res_true = [None for _ in range(len(self.blocks))]
        res_gen = [None for _ in range(len(self.blocks))]
        feature_matrices_true = [None for _ in range(len(self.blocks))]
        feature_matrices_gen = [None for _ in range(len(self.blocks))]

        res_true[0], feature_matrices_true[0] = self.blocks[0](true_wav)
        res_gen[0], feature_matrices_gen[0] = self.blocks[0](gen_wav)

        true_wav = self.pool(true_wav)
        gen_wav = self.pool(gen_wav)

        res_true[1], feature_matrices_true[1] = self.blocks[1](true_wav)
        res_gen[1], feature_matrices_gen[1] = self.blocks[1](gen_wav)

        true_wav = self.pool(true_wav)
        gen_wav = self.pool(gen_wav)

        res_true[2], feature_matrices_true[2] = self.blocks[2](true_wav)
        res_gen[2], feature_matrices_gen[2] = self.blocks[2](gen_wav)

        return res_true, res_gen, \
               feature_matrices_true, feature_matrices_gen
