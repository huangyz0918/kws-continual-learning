import os
import random
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

__classes__ = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]


class SpeechCommandDataset(Dataset):
    def __init__(self, datapath, filename, is_training):
        super(SpeechCommandDataset, self).__init__()
        """
        Args:
            datapath: "./datapath"
            filename: train_filename or valid_filename
            is_training: True or False
        """
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.datapath = datapath
        self.filename = filename
        self.is_training = is_training
        self.class_encoding = {category: index for index, category in enumerate(__classes__)}

        # Load background data as learning noise.
        self.noise_path = os.path.join(self.datapath, "_background_noise_")
        self.noise_dataset = []
        for root, _, filenames in sorted(os.walk(self.noise_path, followlinks=True)):
            for fn in sorted(filenames):
                name, ext = fn.split(".")
                if ext == "wav":
                    self.noise_dataset.append(os.path.join(root, fn))
                    # only add .wav data to noise dataset.

        self.speech_dataset = self.combined_path()

    def combined_path(self):
        dataset_list = []
        for path in self.filename:
            category, wave_name = path.split("/")
            if category in __classes__[:-2]:  # "yes and go"
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, category])
            elif category == "_silence_":
                dataset_list.append(["silence", "silence"])
            else:
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, "unknown"])
        return dataset_list

    def load_audio(self, speech_path):
        waveform, sr = torchaudio.load(speech_path)

        if waveform.shape[1] < self.sample_length:
            # padding if the audio length is smaller than samping length.
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
        else:
            pass

        if self.is_training == True:
            pad_length = int(waveform.shape[1] * 0.1)
            waveform = F.pad(waveform, [pad_length, pad_length])
            offset = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size=(1,)).item()
            waveform = waveform.narrow(1, offset, self.sample_length)

            if self.noise_augmen == True:
                noise_index = torch.randint(0, len(self.noise_dataset), size=(1,)).item()
                noise, noise_sampling_rate = torchaudio.load(self.noise_dataset[noise_index])

                offset = torch.randint(0, noise.shape[1] - self.sample_length + 1, size=(1,)).item()
                noise = noise.narrow(1, offset, self.sample_length)

                background_volume = torch.rand(size=(1,)).item() * 0.1
                waveform.add_(noise.mul_(background_volume)).clamp(-1, 1)
            else:
                pass
        return waveform

    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        self.noise_augmen = self.is_training and random.random() > 0.5

        speech_path = self.speech_dataset[index][0]
        speech_category = self.speech_dataset[index][1]
        label = self.one_hot(speech_category)

        if speech_path == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        return waveform, label
