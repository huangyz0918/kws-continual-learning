"""
Trainer class for KWS.

@author huangyz0918
@date 06/08/2021
"""

import os
import neptune
import numpy as tqdm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataloader import SpeechCommandDataset
from .tc_resnet import TCResNet, STFT_TCResnet, MFCC_TCResnet
from .util import readlines, parameter_number, prepare_device


class Trainer:
    def __init__(self, opt, class_list, model=None):
        self.opt = opt
        self.epoch = opt.epoch
        self.lr = opt.lr
        self.batch = opt.batch
        self.step = opt.step
        self.device, self.device_list = prepare_device(opt.gpu)
        self.class_list = class_list
        self.model = model

        train_filename = readlines(f"{self.opt.dpath}/splits/train.txt")
        valid_filename = readlines(f"{self.opt.dpath}/splits/valid.txt")
        train_dataset = SpeechCommandDataset(f"{self.opt.dpath}/data", train_filename, True, self.class_list)
        valid_dataset = SpeechCommandDataset(f"{self.opt.dpath}/data", valid_filename, False, self.class_list)
        self.templet = "EPOCH: {:01d}  Train: loss {:0.3f}  Acc {:0.2f}  |  Valid: loss {:0.3f}  Acc {:0.2f}"

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch, shuffle=True, drop_last=True)
        self.train_length = len(self.train_dataloader)
        self.valid_length = len(self.valid_dataloader)

        print(">>>   Train length: {}, Valid length: {}, Batch Size: {}".format(self.train_length, self.valid_length, self.batch))
        print(f">>>  Keywords: {self.class_list}")

        if self.model is None:
            if self.opt.model == "stft":
                self.model = STFT_TCResnet(
                    filter_length=256, hop_length=129, bins=129,
                    channels=self.opt.cha, channel_scale=self.opt.scale, num_classes=len(train_dataset.classes)).to(self.device)
            elif self.opt.model == "mfcc":
                self.model = MFCC_TCResnet(
                    bins=40, channel_scale=self.opt.scale, num_classes=len(train_dataset.classes)).to(self.device)
        else:
            self.model.to(self.device)

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
