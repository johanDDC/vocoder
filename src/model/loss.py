import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def G_loss(self, disc_res):
        loss = 0
        for res in disc_res:
            loss += torch.mean((res - 1) ** 2)
        return loss

    def D_loss(self, disc_res_true, disc_res_gen):
        loss = 0
        for i in range(len(disc_res_true)):
            loss += (torch.mean((disc_res_true[i] - 1) ** 2) +
                     torch.mean(disc_res_gen[i] ** 2))
        return loss

    def mel_loss(self, mel_true, mel_gen):
        return F.l1_loss(mel_gen, mel_true)

    def feature_loss(self, feature_matrices_true, feature_matrices_gen):
        loss = 0
        for i in range(len(feature_matrices_true)):
            for j in range(len(feature_matrices_true[i])):
                loss += F.l1_loss(feature_matrices_true[i][j],
                                  feature_matrices_gen[i][j])
        return loss

    def forward(self):
        raise NotImplementedError
