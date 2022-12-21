import glob
import os
import warnings
import random

import librosa
import torchaudio
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval import evaluate
from src.data.dataset import LJSpeechDataset
from src.data.melspecs import MelSpectrogram
from src.model.discriminator.MPD import MPD
from src.model.discriminator.MSD import MSD
from src.model.generator.generator import Generator
from src.model.loss import Loss
from utils import DEVICE, Dict
from ast import literal_eval

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 3407
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
random.seed(SEED)
np.random.seed(SEED)


def train_epoch(model, optimizer, scheduler, train_loader, melspec, train_cfg, log_cfg):
    generator, mpd, msd = model
    generator.train()
    mpd.train()
    msd.train()

    G_opt, D_opt = optimizer
    G_scheduler, D_scheduler = scheduler

    loss_fn = Loss()
    G_losses = []
    D_losses = []
    for step, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
        mels = batch["mels"].to(DEVICE, non_blocking=True)
        wavs = batch["wavs"].unsqueeze(dim=1).to(DEVICE, non_blocking=True)

        G_wavs = generator(mels)
        wavs = torch.nn.functional.pad(wavs, (0, np.abs(G_wavs.shape[-1]) - wavs.shape[-1]))
        G_wavs = torch.nn.functional.pad(G_wavs, (0, np.abs(G_wavs.shape[-1] - wavs.shape[-1])))

        G_mels = melspec(G_wavs.squeeze(dim=1).cpu()).to(DEVICE, non_blocking=True)
        G_mels = torch.nn.functional.pad(G_mels, (0, np.abs(G_mels.shape[-1] - mels.shape[-1])), value=-11.5129251)
        mels = torch.nn.functional.pad(mels, (0, np.abs(G_mels.shape[-1] - mels.shape[-1])), value=-11.5129251)


        D_opt.zero_grad()
        mpd_true, mpd_gen, _, _ = mpd(wavs, G_wavs.detach())
        msd_true, msd_gen, _, _ = msd(wavs, G_wavs.detach())

        loss = loss_fn.D_loss(mpd_true, mpd_gen)
        loss += loss_fn.D_loss(msd_true, msd_gen)
        loss.backward()
        nn.utils.clip_grad_norm(nn.ModuleList([mpd, msd]).parameters(),
                                train_cfg.grad_th_clip)
        D_opt.step()
        D_losses.append(loss.item())

        # generator
        G_opt.zero_grad()
        mpd_res_true, mpd_res_gen, mpd_feature_matrices_true, mpd_feature_matrices_gen \
            = mpd(wavs, G_wavs)
        msd_res_true, msd_res_gen, msd_feature_matrices_true, msd_feature_matrices_gen \
            = msd(wavs, G_wavs)

        feature_loss = loss_fn.feature_loss(mpd_feature_matrices_true, mpd_feature_matrices_gen)
        feature_loss += loss_fn.feature_loss(msd_feature_matrices_true, msd_feature_matrices_gen)
        G_loss = loss_fn.G_loss(mpd_res_gen) + loss_fn.G_loss(msd_res_gen)
        mel_loss = loss_fn.mel_loss(mels, G_mels)

        loss = 45 * mel_loss + 2 * feature_loss + G_loss
        loss.backward()
        nn.utils.clip_grad_norm(generator.parameters(),
                                train_cfg.grad_th_clip)
        G_opt.step()
        G_losses.append(loss.item())

        G_scheduler.step()
        D_scheduler.step()

        if step % log_cfg.steps == 0:
            wandb.log({
                "D_loss": D_losses[-1],
                "mel_loss": mel_loss.item(),
                "feature_loss": feature_loss.item(),
                "G_loss": G_loss.item(),
                "cumulative_G_loss": G_losses[-1]
            })

        if step > 0 and step % log_cfg.log_audio_steps == 0:
            wandb.log({
                "some train audio": wandb.Audio(G_wavs[0].cpu().detach().numpy().T,
                                              sample_rate=22050)
            })
    return np.mean(D_losses), np.mean(G_losses)


def train(cfg, mel_cfg):
    train_cfg = cfg.train_config
    opt_cfg = cfg.optimizer_config
    data_cfg = cfg.data_config
    model_cfg = cfg.model_config
    log_cfg = cfg.log_config

    n_epoches = train_cfg.n_epoches
    batch_size = train_cfg.batch_size

    train_dataset = LJSpeechDataset(data_cfg.data_path, mel_cfg, data_cfg.max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=train_dataset.collator, shuffle=True,
                              num_workers=8, pin_memory=True)

    generator = Generator(model_cfg.generator.in_channels,
                          model_cfg.generator.inner_channels,
                          model_cfg.generator.k_u,
                          model_cfg.generator.k_r,
                          model_cfg.generator.D_r).to(DEVICE)
    mpd = MPD().to(DEVICE)
    msd = MSD().to(DEVICE)

    wandb.watch(generator, log="all", log_freq=10)
    betas = literal_eval(opt_cfg.betas)

    G_opt = torch.optim.AdamW(generator.parameters(), lr=opt_cfg.lr,
                              betas=betas, weight_decay=opt_cfg.wd)
    D_opt = torch.optim.AdamW(nn.ModuleList([mpd, msd]).parameters(), lr=opt_cfg.lr,
                              betas=betas, weight_decay=opt_cfg.wd)

    G_scheduler = torch.optim.lr_scheduler.ExponentialLR(G_opt,
                                                         gamma=opt_cfg.scheduler_gamma)
    D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_opt,
                                                         gamma=opt_cfg.scheduler_gamma)
    melspec = MelSpectrogram(mel_cfg)
    test_files = glob.glob(os.path.join(data_cfg.test_path, "**", "*.wav"), recursive=True)
    os.makedirs(data_cfg.result_path, exist_ok=True)
    os.makedirs(log_cfg.checkpoint_path, exist_ok=True)
    for epoch in range(1, n_epoches + 1):
        D_loss, G_loss = train_epoch((generator, mpd, msd), (G_opt, D_opt),
                                     (G_scheduler, D_scheduler), train_loader,
                                     melspec, train_cfg, log_cfg)

        torch.save({"model_g": generator.state_dict(),
                    "model_mpd": mpd.state_dict(),
                    "model_msd": msd.state_dict(),
                    "opt_g": G_opt.state_dict(),
                    "opt_d": D_opt.state_dict()}, f"{log_cfg.checkpoint_path}/checkpoint_{epoch}")

        wandb_log = {}
        for file in test_files:
            wandb_log[file] = evaluate(generator, file, mel_cfg, data_cfg)

        wandb_log["epoch_D_loss"] = D_loss
        wandb_log["epoch_G_loss"] = G_loss
        wandb.log(wandb_log)


if __name__ == "__main__":
    melspec_cfg = yaml.load(open("./config/mel_config.yaml", "r"), Loader=yaml.FullLoader)
    melspec_cfg = Dict(melspec_cfg).mel_config
    cfg = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    cfg = Dict(cfg).config
    #
    with wandb.init(project="vocoder", entity="johan_ddc_team", config=cfg,
                    name="main_run"):
        train(cfg, melspec_cfg)
