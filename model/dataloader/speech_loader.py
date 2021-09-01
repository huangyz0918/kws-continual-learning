"""
@author huangyz0918
@date 06/08/2021
"""

import os
import random
import librosa
import numpy as np

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .util import snr_noise

class SpeechCommandDataset(Dataset):
    def __init__(self, datapath, filename, is_training, class_list):
        super(SpeechCommandDataset, self).__init__()
        """
        Args:
            datapath: "./datapath"
            filename: train_filename or valid_filename
            is_training: True or False
        """
        self.classes = class_list
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.datapath = datapath
        self.filename = filename
        self.is_training = is_training
        self.class_encoding = {category: index for index, category in enumerate(self.classes)}
        self.speech_dataset = self.combined_path()

    def combined_path(self):
        dataset_list = []
        for path in self.filename:
            category, wave_name = path.split("/")
            if category in self.classes[:-2]:
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
        return waveform

    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        speech_path = self.speech_dataset[index][0]
        speech_category = self.speech_dataset[index][1]
        label = self.one_hot(speech_category)

        if speech_path == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        return waveform, label


class RehearsalDataset(Dataset):
    def __init__(self, datapath, filename, is_training, class_list, replay_class_list, replay_ratio=0.1):
        super(RehearsalDataset, self).__init__()
        """
            Replay the historical data to overcome forgetting issue.
        """
        self.classes = class_list
        self.replay_class_list = replay_class_list
        self.replay_ratio = replay_ratio
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.datapath = datapath
        self.filename = filename
        self.is_training = is_training
        concated_list = self.classes + self.replay_class_list
        self.class_encoding = {category: index for index, category in enumerate([i for j, i in enumerate(concated_list) if i not in (concated_list)[:j]])}
        self.speech_dataset, self.replay_dataset = self.combined_path()

    def combined_path(self):
        dataset_list = []
        replay_data_list = []
        for path in self.filename:
            category, wave_name = path.split("/")

            # load the rehearsal data.
            if category in self.replay_class_list[:-2]:
                path = os.path.join(self.datapath, category, wave_name)
                replay_data_list.append([path, category])
            elif category == "_silence_":
                replay_data_list.append(["silence", "silence"])
            else:
                path = os.path.join(self.datapath, category, wave_name)
                replay_data_list.append([path, "unknown"])

            # load the training data.
            if category in self.classes[:-2]:
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, category])
            elif category == "_silence_":
                dataset_list.append(["silence", "silence"])
            else:
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, "unknown"])
        return dataset_list, replay_data_list

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
        return waveform

    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        train_data_length = len(self.speech_dataset)
        if self.replay_ratio > 0:
            if index % (1 / self.replay_ratio) == 0 and self.is_training:
                speech_path = self.replay_dataset[index][0]
                speech_category = self.replay_dataset[index][1]
            else:
                speech_path = self.speech_dataset[index][0]
                speech_category = self.speech_dataset[index][1]
        else:
            raise ValueError("the replay data ratio should be a number bigger than 0.")
        label = self.one_hot(speech_category)

        if speech_path == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        return waveform, label

class ContinualNoiseDataset(Dataset):
    """
    FIXME:
    Continual learning task protocol using different noise degrees.

    Args:
    datapath: "./datapath"
    filename: train_filename or valid_filename
    is_training: True or False
    class_list: the list of model training keywords
    """
    def __init__(self, datapath, filename, is_training, class_list, noise_type=0, snr_db=10):
        super(ContinualNoiseDataset, self).__init__()
        self.classes = class_list
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.datapath = datapath
        self.filename = filename
        self.noise_type = noise_type
        self.noise_snr = snr_db
        self.is_training = is_training
        self.class_encoding = {category: index for index, category in enumerate(self.classes)}

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
            if category in self.classes[:-2]:
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, category])
            elif category == "_silence_":
                dataset_list.append(["silence", "silence"])
            else:
                path = os.path.join(self.datapath, category, wave_name)
                dataset_list.append([path, "unknown"])
        return dataset_list

    def load_audio(self, speech_path):
        if self.is_training == True:
            return snr_noise(speech_path, self.noise_dataset[self.noise_type],
                                snr_db=self.noise_snr, sample_length=self.sample_length)

    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        speech_path = self.speech_dataset[index][0]
        speech_category = self.speech_dataset[index][1]
        label = self.one_hot(speech_category)

        if speech_path == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        return waveform, label
