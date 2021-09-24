"""
Training script of KWS models using EWC-Online as the CL method.
Reference: Progress & Compress: A scalable framework for continual learning
https://arxiv.org/abs/1805.06370

@author huangyz0918
@date 05/09/2021
"""
import time
import neptune
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import STFT_TCResnet, MFCC_TCResnet, STFT_MLP, MFCC_RNN, parameter_number
from model import Trainer, Evaluator, get_dataloader_keyword
from model.util import get_params, get_gards

# for EWC method to calculate the importance of the weight.
fish = None
cached_checkpoint = None


def on_task_update(model, optimizer, device, train_loader, batch_size, gamma=1):
    fish = torch.zeros_like(get_params(model))
    logsoft = nn.LogSoftmax(dim=1)
    for _, (waveforms, labels) in enumerate(train_loader):
        waveforms, labels = waveforms.to(device), labels.to(device)
        for ex, lab in zip(waveforms, labels):  # EWC Online
            optimizer.zero_grad()
            output = model(ex.unsqueeze(0))
            loss = - F.nll_loss(logsoft(output), lab.unsqueeze(0), reduction='none')
            exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
            loss = torch.mean(loss)
            loss.backward()
            fish += exp_cond_prob * get_gards(model) ** 2

    fish /= (len(train_loader) * batch_size)
    fish *= gamma
    fish += fish

    return fish, get_params(model).data.clone()


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--elambda", default=6, type=float, help="EWC Lambda, the regularization strength")
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--log", default=False, action='store_true',
                            help="record the experiment into web neptune.ai")
        parser.add_argument('--gamma', type=float, default=1, help='gamma parameter for EWC online')
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
        parser.add_argument("--dpath", default="./dataset", type=str, help="The path of dataset")

        parser.add_argument("--model", default="stft", type=str, help="[stft, mfcc]")
        parser.add_argument("--cha", default=config["tc-resnet8"], type=list,
                            help="The channel of model layers (in list)")
        parser.add_argument("--scale", default=1, type=int, help="The scale of model channel")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="stft", type=str, help="The save name")
        args = parser.parse_args()
        return args


    class_list_0 = ["yes", "no", "nine", "three", "bed", "up", "down", "wow", "happy", "four"]
    class_list_1 = ["stop", "go"]
    class_list_2 = ["dog", "cat"]
    class_list_3 = ["two", "bird"]
    class_list_4 = ["eight", "five"]
    class_list_5 = ["tree", "one"]
    class_list_6 = ["left", "right"]
    class_list_7 = ["seven", "six"]
    class_list_8 = ["marvin", "on"]
    class_list_9 = ["off", "house"]
    class_list_10 = ["zero", "sheila"]

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    # initialize and setup Neptune
    if parameters.log:
        neptune.init('huangyz0918/kws')
        neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet', 'EWC-Online'],
                                  params=vars(parameters))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build a multi-head setting for learning process.
    total_class_list = []
    learning_tasks = [class_list_0, class_list_1, class_list_2, class_list_3, class_list_4, class_list_5, class_list_6,
                      class_list_7, class_list_8, class_list_9, class_list_10]
    for x in learning_tasks:
        total_class_list += x
    total_class_num = len([i for j, i in enumerate(total_class_list) if i not in total_class_list[:j]])
    class_list = []
    for task in learning_tasks:
        class_list += task
    class_encoding = {category: index for index, category in enumerate(class_list)}

    # load the model.
    if parameters.model == "stft":
        model = STFT_TCResnet(
            filter_length=256, hop_length=129, bins=129,
            channels=parameters.cha, channel_scale=parameters.scale, num_classes=total_class_num)
    elif parameters.model == "mfcc":
        model = MFCC_TCResnet(bins=40, channels=parameters.cha, channel_scale=parameters.scale,
                              num_classes=total_class_num)
    elif parameters.model == "stft-mlp":
        model = STFT_MLP(filter_length=256, hop_length=129, bins=129, num_classes=total_class_num)
    elif parameters.model == "rnn":
        model = MFCC_RNN(n_mfcc=12, sampling_rate=16000,
                         num_classes=total_class_num)  # sample length for the dataset is 16000.
    else:
        model = None

    # continuous learning by EWC.
    la_list = []
    acc_list = []
    bwt_list = []
    learned_class_list = []
    trainer = Trainer(parameters, model)
    optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr, momentum=0.9)
    start_time = time.time()
    for task_id, task_class in enumerate(learning_tasks):
        print(">>>   Learned Class: ", learned_class_list, " To Learn: ", task_class)
        learned_class_list += task_class
        train_loader, test_loader = get_dataloader_keyword(parameters.dpath, task_class, class_encoding,
                                                           parameters.batch)
        # starting training.
        if parameters.log:
            trainer.ewc_on_train(optimizer, train_loader, test_loader, fish, cached_checkpoint, parameters.elambda,
                                 tag=task_id)
        else:
            trainer.ewc_on_train(optimizer, train_loader, test_loader, fish, cached_checkpoint, parameters.elambda)
        # update the EWC parameters.
        fish, cached_checkpoint = on_task_update(trainer.model, optimizer, device, train_loader, parameters.batch)
        # start evaluating the CL on previous tasks.
        total_learned_acc = 0
        for val_id, task in enumerate(learning_tasks):
            print(f">>>   Testing on task {val_id}, Keywords: {task}")
            _, val_loader = get_dataloader_keyword(parameters.dpath, task, class_encoding, parameters.batch)
            if parameters.log:
                log_data = Evaluator(trainer.model, tag=f't{task_id}v{val_id}').evaluate(val_loader)
            else:
                log_data = Evaluator(trainer.model).evaluate(val_loader)
            if parameters.log:
                neptune.log_metric(f'TASK-{task_id}-acc', log_data["test_accuracy"])
            if val_id <= task_id:
                total_learned_acc += log_data["test_accuracy"]
            if val_id == task_id:
                la_list.append(log_data["test_accuracy"])

        acc_list.append(total_learned_acc / (task_id + 1))
        if task_id > 0:
            bwt_list.append(np.mean([acc_list[i + 1] - acc_list[i] for i in range(len(acc_list) - 1)]))

    duration = time.time() - start_time
    print(f'Training finished, time for {parameters.epoch} epoch: {duration}, average: {duration / parameters.epoch}s')
    print(f'ACC: {np.mean(acc_list)}, std: {np.std(acc_list)}')
    print(f'LA: {np.mean(la_list)}, std: {np.std(la_list)}')
    print(f'BWT: {np.mean(bwt_list)}, std: {np.std(bwt_list)}')
    print(f'Parameter: {parameter_number(trainer.model) / 1024} K')