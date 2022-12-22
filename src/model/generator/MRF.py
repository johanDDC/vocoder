import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, channels, kr, dr):
        super().__init__()
        self.blocks = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kr, dilation=dr[0],
                                  padding=((kr - 1) * dr[0]) // 2)),
            weight_norm(nn.Conv1d(channels, channels, kr, dilation=dr[1],
                                  padding=((kr - 1) * dr[1]) // 2)),
        ])
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(self.relu(x))
        return x


class MRF(nn.Module):
    def __init__(self, channels, kr, dr):
        super().__init__()
        self.kr = kr
        self.blocks = nn.ModuleList([
            ResBlock(channels, kr[i], dr[i]) for i in range(len(kr))
        ])

    def forward(self, x):
        x = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            x = x + self.blocks[i](x)
        return x / len(self.kr)
