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

from .dataloader import SpeechCommandDataset, ContinualNoiseDataset, RehearsalDataset
from .model import TCResNet, STFT_TCResnet, MFCC_TCResnet
from .util import readlines, parameter_number, prepare_device


def get_dataloader_keyword(data_path, class_list, class_encoding, batch_size=1):
    """
    CL task protocol: keyword split.
    To get the GSC data and build the data loader from a list of keywords.
    """
    train_filename = readlines(f"{data_path}/splits/train.txt")
    valid_filename = readlines(f"{data_path}/splits/valid.txt")
    train_dataset = SpeechCommandDataset(f"{data_path}/data", train_filename, True, class_list, class_encoding)
    valid_dataset = SpeechCommandDataset(f"{data_path}/data", valid_filename, False, class_list, class_encoding)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, valid_dataloader


def get_dataloader_replay(data_path, class_list, replay_list, class_encoding, replay_ratio=0.1, batch_size=1):
    """
    CL task protocol: keyword split.
    To get the data mixed with rehearsal data to overcome the catastrophic forgetting.
    """
    train_filename = readlines(f"{data_path}/splits/train.txt")
    valid_filename = readlines(f"{data_path}/splits/valid.txt")
    train_dataset = RehearsalDataset(f"{data_path}/data", train_filename, True, class_list,
                                        class_encoding, replay_list, replay_ratio=replay_ratio)
    valid_dataset = RehearsalDataset(f"{data_path}/data", valid_filename, False, class_list, 
                                        class_encoding, replay_list, replay_ratio=replay_ratio)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, valid_dataloader


def get_dataloader_noise(data_path, class_list, batch_size=1, noise_type=0, snr_db=10):
    """
    CL task protocol: noise permutation.
    To get the GSC data and build the data loader from a list of keywords.
    """
    train_filename = readlines(f"{data_path}/splits/train.txt")
    valid_filename = readlines(f"{data_path}/splits/valid.txt")
    train_dataset = ContinualNoiseDataset(f"{data_path}/data", train_filename, True, class_list, noise_type, snr_db)
    valid_dataset = ContinualNoiseDataset(f"{data_path}/data", valid_filename, False, class_list, noise_type, snr_db)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, valid_dataloader

class Trainer:
    """
    The KWS model training class.
    """
    def __init__(self, opt, class_list, train_dataloader, valid_dataloader, model, tag=None):
        self.opt = opt
        self.tag = tag
        self.epoch = opt.epoch
        self.lr = opt.lr
        self.batch = opt.batch
        self.step = opt.step
        self.class_list = class_list
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device, self.device_list = prepare_device(opt.gpu)

        self.train_length = len(self.train_dataloader)
        self.valid_length = len(self.valid_dataloader)
        self.templet = "EPOCH: {:01d}  Train: loss {:0.3f}  Acc {:0.2f}  |  Valid: loss {:0.3f}  Acc {:0.2f}"

        # map the model weight to the device.
        self.model.to(self.device)

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

            neptune.log_metric(f'{self.tag}-epoch', self.epo)
            neptune.log_metric(f'{self.tag}-train_loss', self.loss_name["train_loss"])
            neptune.log_metric(f'{self.tag}-val_loss', self.loss_name["valid_loss"])
            neptune.log_metric(f'{self.tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
            neptune.log_metric(f'{self.tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])
            
        return self.model

    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if self.loss_name["valid_accuracy"] >= 90.0:
            torch.save(self.mode.state_dict(), os.path.join(save_directory, "best_" + str(self.loss_name["valid_accuracy"]) + ".pt"))

        if (self.epo + 1) % self.opt.freq == 0:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "model" + str(self.epoch + 1) + ".pt"))

        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))


class EWC_Trainer:
    """
    The KWS model training class using EWC as the continual learning method.
    """
    def __init__(self, opt, class_list, train_dataloader, valid_dataloader,
                    task_id, fisher_dict, optpar_dict, ewc_lambda, model, tag=None):
        self.opt = opt
        self.tag = tag
        self.epoch = opt.epoch
        self.lr = opt.lr
        self.batch = opt.batch
        self.step = opt.step
        self.class_list = class_list
        self.model = model
        self.task_id = task_id
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = fisher_dict
        self.optpar_dict = optpar_dict
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device, self.device_list = prepare_device(opt.gpu)

        self.train_length = len(self.train_dataloader)
        self.valid_length = len(self.valid_dataloader)
        self.templet = "EPOCH: {:01d}  Train: loss {:0.3f}  Acc {:0.2f}  |  Valid: loss {:0.3f}  Acc {:0.2f}"

        # map the model weight to the device.
        self.model.to(self.device)

        # enable multi GPU training.
        if len(self.device_list) > 1:
            print(f">>>   Avaliable GPU device: {self.device_list}")
            self.model = nn.DataParallel(self.model)

        self.criterion = F.cross_entropy
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
                
                self.optimizer.zero_grad()
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)

                # calculate the weight improtance (EWC) by adding regularization.
                for t_id in range(self.task_id):
                    for name, param in self.model.named_parameters():
                        fisher = self.fisher_dict[t_id][name]
                        optpar = self.optpar_dict[t_id][name]
                        regular_v = (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda
                        loss += regular_v

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

            neptune.log_metric(f'{self.tag}-epoch', self.epo)
            neptune.log_metric(f'{self.tag}-train_loss', self.loss_name["train_loss"])
            neptune.log_metric(f'{self.tag}-val_loss', self.loss_name["valid_loss"])
            neptune.log_metric(f'{self.tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
            neptune.log_metric(f'{self.tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])
            
        return self.model

    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if self.loss_name["valid_accuracy"] >= 90.0:
            torch.save(self.mode.state_dict(), os.path.join(save_directory, "best_" + str(self.loss_name["valid_accuracy"]) + ".pt"))

        if (self.epo + 1) % self.opt.freq == 0:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "model" + str(self.epoch + 1) + ".pt"))

        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))