"""
Trainer class for KWS.

@author huangyz0918
@date 06/08/2021
"""

import os
import neptune
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataloader import SpeechCommandDataset, ContinualNoiseDataset, RehearsalDataset
from .util import readlines, prepare_device
from .util import get_params, get_gards, store_grad, overwrite_grad, project2cone2, project


def get_dataloader_keyword(data_path, class_list, class_encoding, batch_size=1):
    """
    CL task protocol: keyword split.
    To get the GSC data and build the data loader from a list of keywords.
    """
    if len(class_list) != 0:
        train_filename = readlines(f"{data_path}/splits/train.txt")
        valid_filename = readlines(f"{data_path}/splits/valid.txt")
        train_dataset = SpeechCommandDataset(f"{data_path}/data", train_filename, True, class_list, class_encoding)
        valid_dataset = SpeechCommandDataset(f"{data_path}/data", valid_filename, False, class_list, class_encoding)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_dataloader, valid_dataloader
    else:
        raise ValueError("the class list is empty!")


def get_dataloader_replay(data_path, class_list, replay_list, class_encoding, replay_ratio=0.1, batch_size=1):
    """
    CL task protocol: keyword split.
    To get the data mixed with rehearsal data to overcome the catastrophic forgetting.
    """
    if len(class_list) != 0:
        train_filename = readlines(f"{data_path}/splits/train.txt")
        valid_filename = readlines(f"{data_path}/splits/valid.txt")
        train_dataset = RehearsalDataset(f"{data_path}/data", train_filename, True, class_list,
                                         class_encoding, replay_list, replay_ratio=replay_ratio)
        valid_dataset = RehearsalDataset(f"{data_path}/data", valid_filename, False, class_list,
                                         class_encoding, replay_list, replay_ratio=replay_ratio)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_dataloader, valid_dataloader
    else:
        raise ValueError("the class list is empty!")


def get_dataloader_noise(data_path, class_list, batch_size=1, noise_type=0, snr_db=10):
    """
    CL task protocol: noise permutation.
    To get the GSC data and build the data loader from a list of keywords.
    """
    if len(class_list) != 0:
        train_filename = readlines(f"{data_path}/splits/train.txt")
        valid_filename = readlines(f"{data_path}/splits/valid.txt")
        train_dataset = ContinualNoiseDataset(f"{data_path}/data", train_filename, True, class_list, noise_type, snr_db)
        valid_dataset = ContinualNoiseDataset(f"{data_path}/data", valid_filename, False, class_list, noise_type,
                                              snr_db)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_dataloader, valid_dataloader
    else:
        raise ValueError("the class list is empty!")


class Trainer:
    """
    The KWS model training class.
    """

    def __init__(self, opt, model):
        self.opt = opt
        self.lr = opt.lr
        self.step = opt.step
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.model = model
        self.device, self.device_list = prepare_device(opt.gpu)
        self.templet = "EPOCH: {:01d}  Train: loss {:0.3f}  Acc {:0.2f}  |  Valid: loss {:0.3f}  Acc {:0.2f}"
        # map the model weight to the device.
        self.model.to(self.device)
        # enable multi GPU training.
        if len(self.device_list) > 1:
            print(f">>>   Avaliable GPU device: {self.device_list}")
            self.model = nn.DataParallel(self.model)

        self.criterion = nn.CrossEntropyLoss()
        self.loss_name = {
            "train_loss": 0.0, "train_accuracy": 0.0, "train_total": 0, "train_correct": 0,
            "valid_loss": 0.0, "valid_accuracy": 0.0, "valid_total": 0, "valid_correct": 0}

    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        if self.loss_name["valid_accuracy"] >= 90.0:
            torch.save(self.model.state_dict(),
                       os.path.join(save_directory, "best_" + str(self.loss_name["valid_accuracy"]) + ".pt"))

        if (self.epo + 1) % self.opt.freq == 0:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "model" + str(self.epoch + 1) + ".pt"))

        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))

    def model_train(self, task_id, optimizer, train_dataloader, valid_dataloader, is_pnn=False, tag=None):
        """
        Normal model training process, without modifing the loss function.
        """
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                if is_pnn:
                    logits = self.model(waveform, task_id)
                else:
                    logits = self.model(waveform)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    if is_pnn:
                        logits = self.model(waveform, task_id)
                    else:
                        logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]

            scheduler.step()
            self.model_save()
            print(
                self.templet.format(self.epo + 1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"],
                                    self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))

            if tag:
                neptune.log_metric(f'{tag}-epoch', self.epo)
                neptune.log_metric(f'{tag}-train_loss', self.loss_name["train_loss"])
                neptune.log_metric(f'{tag}-val_loss', self.loss_name["valid_loss"])
                neptune.log_metric(f'{tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
                neptune.log_metric(f'{tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])

    def ewc_train(self, task_id, optimizer, train_dataloader, valid_dataloader,
                  fisher_dict, optpar_dict, ewc_lambda, tag=None):
        """
        Using Elastic Weight Consolidation (EWC) as the continual learning method.

        @article{kirkpatrick2017overcoming,
            title={Overcoming catastrophic forgetting in neural networks},
            author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
            journal={Proceedings of the national academy of sciences},
            volume={114},
            number={13},
            pages={3521--3526},
            year={2017},
            publisher={National Acad Sciences}
            }
        """
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)

                # calculate the weight improtance (EWC) by adding regularization.
                for t_id in range(task_id):
                    for name, param in self.model.named_parameters():
                        fisher = fisher_dict[t_id][name]
                        optpar = optpar_dict[t_id][name]
                        loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

                loss.backward()
                optimizer.step()

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]

            scheduler.step()
            self.model_save()
            print(
                self.templet.format(self.epo + 1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"],
                                    self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))

            if tag:
                neptune.log_metric(f'{tag}-epoch', self.epo)
                neptune.log_metric(f'{tag}-train_loss', self.loss_name["train_loss"])
                neptune.log_metric(f'{tag}-val_loss', self.loss_name["valid_loss"])
                neptune.log_metric(f'{tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
                neptune.log_metric(f'{tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])

    def si_train(self, optimizer, train_dataloader, valid_dataloader,
                 big_omega, small_omega, cached_checkpoint, coefficient=1, tag=None):
        """
        Using Synaptic Intelligence (SI) as the continual learning method.

        @inproceedings{zenke2017continual,
            title={Continual Learning Through Synaptic Intelligence},
            author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
            booktitle={International Conference on Machine Learning},
            year={2017},
            url={https://arxiv.org/abs/1703.04200}
        }
        """
        updated_small_omega = small_omega
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        for self.epo in range(self.epoch):
            self.model.train()
            self.loss_name.update({key: 0 for key in self.loss_name})
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(waveform)

                # calculate the loss penalty.
                if big_omega is None:
                    penalty = torch.tensor(0.0).to(self.device)
                else:
                    penalty = (big_omega * ((get_params(self.model) - cached_checkpoint) ** 2)).sum() * coefficient

                loss = self.criterion(logits, labels)
                loss += penalty
                # debug
                # print("big_omega: ", big_omega, " small_omega: ", updated_small_omega, " penalty: ", penalty)
                loss.backward()
                nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 1)
                optimizer.step()

                # update the small_omega value.
                updated_small_omega += self.lr * get_gards(self.model).data ** 2

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]

            scheduler.step()
            self.model_save()
            print(
                self.templet.format(self.epo + 1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"],
                                    self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))

            if tag:
                neptune.log_metric(f'{tag}-epoch', self.epo)
                neptune.log_metric(f'{tag}-train_loss', self.loss_name["train_loss"])
                neptune.log_metric(f'{tag}-val_loss', self.loss_name["valid_loss"])
                neptune.log_metric(f'{tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
                neptune.log_metric(f'{tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])

        return updated_small_omega

    def gem_train(self, optimizer, train_dataloader, valid_dataloader, buffer,
                  grad_dims, grads_cs, grads_da, gamma, tag=None):
        """
        Using Gradient Episodic Memory for Continual Learning (GEM) as the continual learning method.

        @article{lopez2017gradient,
            title={Gradient episodic memory for continual learning},
            author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
            journal={Advances in neural information processing systems},
            volume={30},
            pages={6467--6476},
            year={2017}
            }
        """
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)

                # get the rehearsal data.
                if not buffer.is_empty():
                    buf_inputs, buf_labels, buf_task_labels = buffer.get_data(buffer.buffer_size)

                    for tt in buf_task_labels.unique():
                        # compute gradient on the memory buffer.
                        optimizer.zero_grad()
                        cur_task_inputs = buf_inputs[buf_task_labels == tt]
                        cur_task_labels = buf_labels[buf_task_labels == tt]
                        cur_task_outputs = self.model.forward(cur_task_inputs)
                        penalty = self.criterion(cur_task_outputs, cur_task_labels)
                        penalty.backward()
                        store_grad(self.model.parameters(), grads_cs[tt], grad_dims)

                # now compute the grad on the current data.
                optimizer.zero_grad()
                logits = self.model(waveform)
                loss = self.criterion(logits, labels)
                loss.backward()

                # check if gradient violates buffer constraints.
                if not buffer.is_empty():
                    # copy gradient.
                    store_grad(self.model.parameters(), grads_da, grad_dims)
                    dot_prod = torch.mm(grads_da.unsqueeze(0), torch.stack(grads_cs).T)
                    if (dot_prod < 0).sum() != 0:
                        project2cone2(grads_da.unsqueeze(1), torch.stack(grads_cs).T, margin=gamma)
                        # copy gradients back.
                        overwrite_grad(self.model.parameters(), grads_da, grad_dims)
                optimizer.step()

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]

            scheduler.step()
            self.model_save()
            print(
                self.templet.format(self.epo + 1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"],
                                    self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))

            if tag:
                neptune.log_metric(f'{tag}-epoch', self.epo)
                neptune.log_metric(f'{tag}-train_loss', self.loss_name["train_loss"])
                neptune.log_metric(f'{tag}-val_loss', self.loss_name["valid_loss"])
                neptune.log_metric(f'{tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
                neptune.log_metric(f'{tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])

    def agem_train(self, optimizer, train_dataloader, valid_dataloader, buffer,
                   grad_dims, grad_xy, grad_er, tag=None):
        """
        Using Gradient Episodic Memory for Continual Learning (GEM) as the continual learning method.

        @article{lopez2017gradient,
            title={Gradient episodic memory for continual learning},
            author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
            journal={Advances in neural information processing systems},
            volume={30},
            pages={6467--6476},
            year={2017}
            }
        """
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step, gamma=0.1, last_epoch=-1)
        for self.epo in range(self.epoch):
            self.loss_name.update({key: 0 for key in self.loss_name})
            self.model.train()
            for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader)):
                waveform, labels = waveform.to(self.device), labels.to(self.device)

                # compute the grad on the current data.
                logits = self.model(waveform)
                optimizer.zero_grad()
                loss = self.criterion(logits, labels)
                loss.backward()

                # get the rehearsal data.
                if not buffer.is_empty():
                    store_grad(self.model.parameters(), grad_xy, grad_dims)

                    buf_inputs, buf_labels = buffer.get_data(self.batch)
                    optimizer.zero_grad()
                    buf_outputs = self.model.forward(buf_inputs)
                    penalty = self.criterion(buf_outputs, buf_labels)
                    penalty.backward()
                    store_grad(self.model.parameters(), grad_er, grad_dims)

                    dot_prod = torch.dot(grad_xy, grad_er)
                    if dot_prod.item() < 0:
                        g_tilde = project(gxy=grad_xy, ger=grad_er)
                        overwrite_grad(self.model.parameters(), g_tilde, grad_dims)
                    else:
                        overwrite_grad(self.model.parameters(), grad_xy, grad_dims)

                optimizer.step()

                self.loss_name["train_loss"] += loss.item() / train_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["train_total"] += labels.size(0)
                self.loss_name["train_correct"] += (predict == labels).sum().item()
                self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

            self.model.eval()
            for batch_idx, (waveform, labels) in tqdm(enumerate(valid_dataloader)):
                with torch.no_grad():
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logits = self.model(waveform)
                    loss = self.criterion(logits, labels)

                    self.loss_name["valid_loss"] += loss.item() / valid_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["valid_total"] += labels.size(0)
                    self.loss_name["valid_correct"] += (predict == labels).sum().item()
                    self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]

            scheduler.step()
            self.model_save()
            print(
                self.templet.format(self.epo + 1, self.loss_name["train_loss"], 100 * self.loss_name["train_accuracy"],
                                    self.loss_name["valid_loss"], 100 * self.loss_name["valid_accuracy"]))

            if tag:
                neptune.log_metric(f'{tag}-epoch', self.epo)
                neptune.log_metric(f'{tag}-train_loss', self.loss_name["train_loss"])
                neptune.log_metric(f'{tag}-val_loss', self.loss_name["valid_loss"])
                neptune.log_metric(f'{tag}-train_accuracy', 100 * self.loss_name["train_accuracy"])
                neptune.log_metric(f'{tag}-valid_accuracy', 100 * self.loss_name["valid_accuracy"])
