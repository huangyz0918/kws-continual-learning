import os
import neptune
import librosa
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from dataloader import *
from model import *
from util import *


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.epoch = opt.epoch
        self.lr = opt.lr
        self.batch = opt.batch
        self.step = opt.step
        self.device, self.device_list = prepare_device(opt.gpu)

        train_filename = readlines("./dataset/splits/{}.txt".format("train"))
        valid_filename = readlines("./dataset/splits/{}.txt".format("valid"))
        train_dataset = SpeechCommandDataset("./dataset/data", train_filename, True)
        valid_dataset = SpeechCommandDataset("./dataset/data", valid_filename, False)
        self.templet = "EPOCH: {:01d}  Train: loss {:0.3f}  Acc {:0.2f}  |  Valid: loss {:0.3f}  Acc {:0.2f}"

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        self.train_length = len(self.train_dataloader)
        self.valid_length = len(self.valid_dataloader)
        print(">>>   Train length: {}, Valid length: {}, Batch Size: {}".format(self.train_length, self.valid_length,
                                                                                self.batch))

        if self.opt.model == "stft":
            self.model = STFT_TCResnet(
                filter_length=256, hop_length=129, bins=129,
                channels=self.opt.cha, channel_scale=self.opt.scale, num_classes=12).to(self.device)
        elif self.opt.model == "mfcc":
            self.model = MFCC_TCResnet(
                bins=40, channel_scale=self.opt.scale, num_classes=12).to(self.device)

        print(f">>>   Num of model parameters: {parameter_number(self.model)}")

        # enable multi GPU training.
        if len(self.device_list) > 1:
            print(f">>>   Avaliable GPU device: {self.device_list}")
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        self.loss_name = {
            "train_loss": 0, "train_accuracy": 0, "train_total": 0, "train_correct": 0,
            "valid_loss": 0, "valid_accuracy": 0, "valid_total": 0, "valid_correct": 0}

    def model_train(self):
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(self.train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                logits = self.model(waveform)

                self.optimizer.zero_grad()
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                self.loss_name["train_loss"] += loss.item() / self.train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(self.valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / self.valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]

            self.scheduler.step()
            self.model_save()
            # If you have the logger, you can remove the print function here.
            print(
                self.templet.format(self.epo + 1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"],
                                    self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))

            neptune.log_metric('epoch', self.epo)
            neptune.log_metric('train_loss', self.loss_name["train_loss"])
            neptune.log_metric('val_loss', self.loss_name["valid_loss"])
            neptune.log_metric('train_accuracy', 100 * self.loss_name["train_accuracy"])
            neptune.log_metric('valid_accuracy', 100 * self.loss_name["valid_accuracy"])

    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if self.loss_name["valid_accuracy"] >= 92.0:
            torch.save(self.mode.state_dict(),
                       os.path.join(save_directory, "best_" + str(self.loss_name["valid_accuracy"]) + ".pt"))

        if (self.epo + 1) % self.opt.freq == 0:
            torch.save(self.model.state_dict(),
                       os.path.join(save_directory, "model" + str(self.epoch + 1) + ".pt"))

        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=5, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")

        parser.add_argument("--model", default="stft", type=str, help=["stft", "mfcc"])
        parser.add_argument("--cha", default=config["tc-resnet8"], type=list,
                            help="the channel of model layers (in list)")
        parser.add_argument("--scale", default=3, type=int, help="the scale of model channel")
        parser.add_argument("--freq", default=30, type=int, help="model saving frequency (in step)")
        parser.add_argument("--save", default="stft", type=str, help="the save name")
        args = parser.parse_args()
        return args


    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    # initialize and setup Neptune
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model',
                              tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet'],
                              params=vars(options(config)))

    Trainer(options(config)).model_train()
