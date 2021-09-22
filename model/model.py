"""
The TC-ResNet model.

@author huangyz0918
@date 06/08/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torchaudio.transforms import MFCC

from .util import STFT


class RNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512, n_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        batch_size, _, n_mfcc, _ = inputs.shape
        inputs = inputs.reshape(batch_size, -1, n_mfcc)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        out, _ = self.lstm(inputs, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MLP(nn.Module):
    def __init__(self, hop_length, audio_time, n_class):
        super().__init__()

        self.input_fc = nn.Linear(hop_length * audio_time, 512)
        self.hidden_fc = nn.Linear(512, 100)
        self.output_fc = nn.Linear(100, n_class)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(inputs))
        h_2 = F.relu(self.hidden_fc(h_1))
        out = self.output_fc(h_2)
        return out


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        if in_channels != out_channels:
            stride = 2
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            stride = 1
            self.residual = nn.Sequential()

        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 9), stride=stride, padding=(0, 4), bias=False)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 9), stride=stride, padding=(0, 4), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        res = self.residual(inputs)
        out = self.relu(out + res)
        return out


class TCResNet(nn.Module):
    def __init__(self, bins, n_channels, n_class):
        super(TCResNet, self).__init__()
        """
        Args:
            bin: frequency bin or feature bin
        """
        self.conv = nn.Conv2d(bins, n_channels[0], kernel_size=(1, 3), padding=(0, 1), bias=False)

        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            layers.append(Residual(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(n_channels[-1], n_class)

    def forward(self, inputs):
        """
        Args:
            input
            [B, 1, H, W] ~ [B, 1, freq, time]
            reshape -> [B, freq, 1, time]
        """
        B, C, H, W = inputs.shape
        inputs = rearrange(inputs, "b c f t -> b f c t", c=C, f=H)
        out = self.conv(inputs)
        out = self.layers(out)

        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


class STFT_TCResnet(nn.Module):
    def __init__(self, filter_length, hop_length, bins, channels, channel_scale, num_classes):
        super(STFT_TCResnet, self).__init__()
        sampling_rate = 16000
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.bins = bins
        self.channels = channels
        self.channel_scale = channel_scale
        self.num_classes = num_classes

        self.stft_layer = STFT(self.filter_length, self.hop_length)
        self.tc_resnet = TCResNet(self.bins, [int(cha * self.channel_scale) for cha in self.channels], self.num_classes)

    def __spectrogram__(self, real, imag):
        spectrogram = torch.sqrt(real ** 2 + imag ** 2)
        return spectrogram

    def forward(self, waveform):
        real, imag = self.stft_layer(waveform)
        spectrogram = self.__spectrogram__(real, imag)
        logits = self.tc_resnet(spectrogram)
        return logits


class MFCC_TCResnet(nn.Module):
    def __init__(self, bins: int, channels, channel_scale: int, num_classes=12):
        super(MFCC_TCResnet, self).__init__()
        self.sampling_rate = 16000
        self.bins = bins
        self.channels = channels
        self.channel_scale = channel_scale
        self.num_classes = num_classes

        self.mfcc_layer = MFCC(sample_rate=self.sampling_rate, n_mfcc=self.bins, log_mels=True)
        self.tc_resnet = TCResNet(self.bins, [int(cha * self.channel_scale) for cha in self.channels], self.num_classes)

    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        logits = self.tc_resnet(mel_sepctogram)
        return logits


class STFT_MLP(nn.Module):
    def __init__(self, filter_length, hop_length, bins, num_classes):
        super(STFT_MLP, self).__init__()
        sampling_rate = 16000
        self.bins = bins
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.num_classes = num_classes

        self.stft_layer = STFT(self.filter_length, self.hop_length)
        self.mlp = MLP(self.bins, 125, self.num_classes)

    def __spectrogram__(self, real, imag):
        spectrogram = torch.sqrt(real ** 2 + imag ** 2)
        return spectrogram

    def forward(self, waveform):
        real, imag = self.stft_layer(waveform)
        spectrogram = self.__spectrogram__(real, imag)
        logits = self.mlp(spectrogram)
        return logits


class MFCC_RNN(nn.Module):
    def __init__(self, n_mfcc, sampling_rate, n_layers=1, hidden_size=512, num_classes=12):
        super(MFCC_RNN, self).__init__()
        self.sampling_rate = sampling_rate
        self.num_classes = num_classes
        self.n_mfcc = n_mfcc  # feature length

        self.mfcc_layer = MFCC(sample_rate=self.sampling_rate, n_mfcc=self.n_mfcc, log_mels=True)
        self.rnn = RNN(self.n_mfcc, self.num_classes, hidden_size=hidden_size, n_layers=n_layers)

    def forward(self, waveform):
        mel_sepctogram = self.mfcc_layer(waveform)
        logits = self.rnn(mel_sepctogram)
        return logits


class MLP_PNN(nn.Module):
    """
    Basic PNN network structure.
    """

    def __init__(self, filter_length, hop_length, input_size):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.input_size = input_size
        self.cols = nn.ModuleList()
        self.stft_layer = STFT(self.filter_length, self.hop_length)
        self.hsize_list = []

    def __spectrogram__(self, real, imag):
        spectrogram = torch.sqrt(real ** 2 + imag ** 2)
        return spectrogram

    def add_column(self, num_class, hsize=128):
        for col in self.cols:
            col.freeze()  # freeze all previous columns.

        col_id = len(self.cols)  # create new column.
        self.hsize_list.append(hsize)
        col = MLP_Column(self.input_size, num_class, self.hsize_list, col_id=col_id)
        self.cols.append(col)

    def forward(self, waveform, task_id, lateral_weights=None):
        if lateral_weights is None:
            lateral_weights = [1 for _ in range(task_id)]

        col = self.cols[task_id]
        real, imag = self.stft_layer(waveform)
        spectrogram = self.__spectrogram__(real, imag)
        return col(spectrogram, self.cols[:task_id], lateral_weights)


class MLP_Column(nn.Module):
    """
    The columns of each learning tasks.
    In the code we use a hidden layer of size 128 for task#1 and 32 for all the subsequent tasks.
    """

    def __init__(self, input_size, num_class, hsize_list, col_id=0):
        super().__init__()
        self.hsize_list = hsize_list
        self.l1 = nn.Linear(input_size, self.hsize_list[-1])
        self.l2 = nn.Linear(self.hsize_list[-1], num_class)
        self.Us = nn.ModuleList()
        self.col_id = col_id

        for size in self.hsize_list:
            lateral = nn.Linear(size, num_class)
            self.Us.append(lateral)

    def add_lateral(self, x, prev_cols, lateral_weights):
        outputs = torch.zeros_like(x)
        for col_id, col in enumerate(prev_cols):
            input = col.outputs
            if lateral_weights:
                input = input * lateral_weights[col_id]
            layer = self.Us[col_id]
            outputs += layer(input)
        return outputs

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, prev_cols, lateral_weights):
        for i, col in enumerate(prev_cols):
            l_w = lateral_weights[:i + 1]
            col(x, prev_cols[:i], l_w)

        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))

        self.outputs = x

        x = self.l2(x)
        x += self.add_lateral(x, prev_cols=prev_cols, lateral_weights=lateral_weights)
        return x


class Res_Column(nn.Module):
    """
    The TC-ResNet columns of each learning tasks.
    """

    def __init__(self, num_class, bins, n_channels, col_id=0):
        super().__init__()
        self.conv = nn.Conv2d(bins, n_channels[0], kernel_size=(1, 3), padding=(0, 1), bias=False)
        layers = []
        for in_channels, out_channels in zip(n_channels[0:-1], n_channels[1:]):
            layers.append(Residual(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(n_channels[-1], num_class)

        self.Us = nn.ModuleList()
        self.col_id = col_id

        for _ in range(self.col_id):
            lateral = nn.Linear(n_channels[-1], num_class)
            self.Us.append(lateral)

    def add_lateral(self, inputs, prev_cols, lateral_weights):
        outputs = torch.zeros_like(inputs)
        for col_id, col in enumerate(prev_cols):
            input = col.outputs
            if lateral_weights:
                input = input * lateral_weights[col_id]
            layer = self.Us[col_id]
            outputs += layer(input)
        return outputs

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs, prev_cols, lateral_weights):
        for i, col in enumerate(prev_cols):
            l_w = lateral_weights[:i + 1]
            col(inputs, prev_cols[:i], l_w)

        B, C, H, W = inputs.shape
        inputs = rearrange(inputs, "b c f t -> b f c t", c=C, f=H)
        out = self.conv(inputs)
        out = self.layers(out)

        out = self.pool(out)
        out = out.view(out.shape[0], -1)

        self.outputs = out

        out = self.linear(out)
        out += self.add_lateral(out, prev_cols=prev_cols, lateral_weights=lateral_weights)
        return out


class TC_PNN(nn.Module):
    def __init__(self, bins, filter_length, hop_length):
        super(TC_PNN, self).__init__()
        self.bins = bins
        self.cols = nn.ModuleList()
        self.stft_layer = STFT(filter_length, hop_length)

    def __spectrogram__(self, real, imag):
        spectrogram = torch.sqrt(real ** 2 + imag ** 2)
        return spectrogram

    def add_column(self, num_class, n_channels, channel_scale):
        for col in self.cols:
            col.freeze()  # freeze all previous columns.

        col_id = len(self.cols)  # create new column.
        col = Res_Column(num_class, self.bins, [int(cha * channel_scale) for cha in n_channels], col_id)
        self.cols.append(col)

    def forward(self, waveform, task_id, lateral_weights=None):
        if lateral_weights is None:
            lateral_weights = [1 for _ in range(task_id)]

        col = self.cols[task_id]
        real, imag = self.stft_layer(waveform)
        spectrogram = self.__spectrogram__(real, imag)
        return col(spectrogram, self.cols[:task_id], lateral_weights)
