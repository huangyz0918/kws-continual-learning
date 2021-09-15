"""
Training script of KWS models using GEM as the CL method.
Reference: Gradient Episodic Memory for Continual Learning
Gradient Episodic Memory for Continual Learning

@author huangyz0918
@date 05/09/2021
"""

import neptune
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TCResNet, STFT_TCResnet, MFCC_TCResnet, STFT_MLP, STFT_RNN
from model import Trainer, Evaluator, get_dataloader_keyword
from model.util import Buffer, get_grad_dim


def on_task_update(task_id, task_num, grads_cs, grad_dims, buffer, device, loader):
    """
    Update the regularization after each task learning.
    """
    current_task = task_id + 1
    grads_cs.append(torch.zeros(np.sum(grad_dims)).to(device))

    # add data to the buffer.
    samples_per_task = buffer.buffer_size // task_num
    cur_x, cur_y = next(iter(loader))
    buffer.add_data(
        examples=cur_x.to(device),
        labels=cur_y.to(device),
        task_labels=torch.ones(samples_per_task, dtype=torch.long).to(device) * (current_task - 1)
    )


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        # should be a multiple of batch size.
        parser.add_argument("--bsize", default=12800, type=float, help="the rehearsal buffer size") 
        parser.add_argument('--gamma', type=float, default=0.5, help='Margin parameter for GEM.')
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
        parser.add_argument("--dpath", default="./dataset", type=str, help="The path of dataset")

        parser.add_argument("--model", default="stft", type=str, help=["stft", "mfcc"])
        parser.add_argument("--cha", default=config["tc-resnet8"], type=list,
                            help="The channel of model layers (in list)")
        parser.add_argument("--scale", default=3, type=int, help="The scale of model channel")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="stft", type=str, help="The save name")
        args = parser.parse_args()
        return args

    class_list_1 = ["yes", "no", "nine", "three", "bed", 
                    "up", "down", "wow", "happy", "four",
                    "left", "right", "seven", "six", "marvin", 
                    "on", "off", "house", "zero", "sheila"]
    class_list_2 = ["stop", "go"]
    class_list_3 = ["dog", "cat"]
    class_list_4 = ["two", "bird"]
    class_list_5 = ["eight", "five"]
    class_list_6 = ["tree", "one"]

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    # initialize and setup Neptune
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet', 'GEM'], params=vars(parameters))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build a multi-head setting for learning process.
    total_class_list = []
    learning_tasks = [class_list_1, class_list_2, class_list_3, class_list_4, class_list_5, class_list_6]
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
        model = MFCC_TCResnet(bins=40, channel_scale=parameters.scale, num_classes=total_class_num)
    elif parameters.model == "stft-mlp":
        model = STFT_MLP(filter_length=256, hop_length=129, bins=129, num_classes=total_class_num)
    elif parameters.model == "stft-rnn":
        model = STFT_RNN(filter_length=256, hop_length=129, bins=129, num_classes=len(class_list), hidden_size=512)
    else:
        model = None

    # continuous learning by GEM.
    learned_class_list = []
    trainer = Trainer(parameters, model)
    gem_buffer = Buffer(parameters.bsize, trainer.device)
    # Allocate temporary synaptic memory.
    grad_dims = get_grad_dim(trainer.model)
    grads_cs = []
    grads_da = torch.zeros(np.sum(grad_dims)).to(trainer.device)
    # start continual learning process.
    for task_id, task_class in enumerate(learning_tasks):
        print(">>>   Learned Class: ", learned_class_list, " To Learn: ", task_class)
        learned_class_list += task_class
        train_loader, test_loader = get_dataloader_keyword(parameters.dpath, task_class, class_encoding, parameters.batch)
        # starting training.
        trainer.gem_train(task_id, train_loader, test_loader, gem_buffer, grad_dims, 
                            grads_cs, grads_da, parameters.gamma, tag=task_id)
        # update the GEM parameters.
        on_task_update(task_id, len(learning_tasks), grads_cs, grad_dims, gem_buffer, trainer.device, train_loader)
        # start evaluating the CL on previous tasks.
        total_acc = 0
        for val_id, task in enumerate(learning_tasks):
            print(f">>>   Testing on task {val_id}, Keywords: {task}")
            _, val_loader = get_dataloader_keyword(parameters.dpath, task, class_encoding, parameters.batch)
            log_data = Evaluator(trainer.model, tag=f't{task_id}v{val_id}').evaluate(val_loader)
            neptune.log_metric(f'TASK-{task_id}-acc', log_data["test_accuracy"])
            total_acc += log_data["test_accuracy"]
        print(f">>>   Average Accuracy: {total_acc / len(learning_tasks) * 100}")