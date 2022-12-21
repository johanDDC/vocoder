import glob
import os

import librosa
import torchaudio
import yaml

import wandb
import torch

from src.data.melspecs import MelSpectrogram
from src.model.generator.generator import Generator
from utils import DEVICE, Dict


def evaluate(generator, file, mel_cfg, data_cfg):
    melspec = MelSpectrogram(mel_cfg)
    generator.eval()
    with torch.no_grad():
        test_wav, _ = librosa.load(f"{file}", sr=mel_cfg.sample_rate)
        test_wav = torch.tensor(test_wav)
        test_mel = melspec(test_wav.unsqueeze(0)).squeeze(0)
        gen_wav = generator(test_mel.to(DEVICE))
        wandb_gen = wandb.Audio(gen_wav.cpu().numpy().T,
                                      sample_rate=mel_cfg.sample_rate)
        torchaudio.save(f"{data_cfg.result_path}/{file[file.rfind('/') + 1 :]}",
                        gen_wav.cpu(), sample_rate=mel_cfg.sample_rate)
    return wandb_gen


if __name__ == '__main__':
    cfg = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    cfg = Dict(cfg).config
    melspec_cfg = yaml.load(open("./config/mel_config.yaml", "r"), Loader=yaml.FullLoader)
    melspec_cfg = Dict(melspec_cfg).mel_config

    try:
        model_cfg = cfg.model_config
        generator = Generator(model_cfg.generator.in_channels,
                              model_cfg.generator.inner_channels,
                              model_cfg.generator.k_u,
                              model_cfg.generator.k_r,
                              model_cfg.generator.D_r).to(DEVICE)

        generator.load_state_dict(torch.load(cfg.inference.model_path)["model_g"])
        generator.to(DEVICE)
    except:
        print("You should first train your model")
        exit(1)

    test_files = glob.glob(os.path.join(cfg.data_config.test_path, "**", "*.wav"), recursive=True)
    for file in test_files:
        evaluate(generator, file, melspec_cfg, cfg.data_config)
