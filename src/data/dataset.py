import torch
import torchaudio

from torch.nn.utils.rnn import pad_sequence

from src.data.melspecs import MelSpectrogram


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, data_path, mel_config, max_len=8192):
        super().__init__(root=data_path)
        self.mel_spec = MelSpectrogram(mel_config)
        self.max_len = max_len
        self.mel_cfg = mel_config

    def __getitem__(self, index: int):
        wav, _, _, _ = super().__getitem__(index)
        return wav

    def collator(self, batch):
        wavs = []
        mels = []

        for wav in batch:
            if wav.shape[0] >= self.max_len:
                segments = torch.split(wav, self.max_len, dim=1)
                wav = segments[torch.randint(0, len(segments))]
            wavs.append(wav.squeeze(0))
            melspec = self.mel_spec(wav)
            mels.append(melspec.squeeze(0).transpose(-1, -2))

            #
            # for segment in segments:
            #     if segment.shape[-1] >= self.max_len // 2:
            #         melspec = self.mel_spec(segment)
            #         texts.append(text)
            #

        wavs = pad_sequence(wavs, batch_first=True, padding_value=0)
        mels = pad_sequence(mels, batch_first=True,
                            padding_value=self.mel_cfg.pad_value).transpose(-1, -2)
        return {
            "wavs": wavs,
            "mels": mels
        }
