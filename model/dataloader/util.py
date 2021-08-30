import math

import torch
import torchaudio
import torch.nn.functional as F


def snr_noise(speech_dpath, noise_dpath, snr_db=10, sample_length=16000):
    """
    Add the SNR noise to a speech data.
    """

    # load the speech and nosie data
    waveform, speech_sampling_rate = torchaudio.load(speech_dpath)
    noise, noise_sampling_rate = torchaudio.load(noise_dpath)
    if waveform.shape[1] < sample_length:
        # padding if the audio length is smaller than samping length.
        waveform = F.pad(waveform, [0, sample_length - waveform.shape[1]])
    else:
        pass

    # process the speech audio data
    pad_length = int(waveform.shape[1] * 0.1)
    waveform = F.pad(waveform, [pad_length, pad_length])
    offset = torch.randint(0, waveform.shape[1] - sample_length + 1, size=(1,)).item()
    waveform = waveform.narrow(1, offset, sample_length)

    # process the noise data
    offset = torch.randint(0, noise.shape[1] - sample_length + 1, size=(1,)).item()
    noise = noise.narrow(1, offset, sample_length)

    # add the background noise by SNR
    speech_power = waveform.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * waveform + noise) / 2
    return noisy_speech
