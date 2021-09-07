"""
Training script of KWS models sing SI as the CL method.
Reference: Continual Learning Through Synaptic Intelligence
https://arxiv.org/abs/1703.04200

@author huangyz0918
@date 05/09/2021
"""

import neptune
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TCResNet, STFT_TCResnet, MFCC_TCResnet, STFT_MLP
from model import Trainer, Evaluator, get_dataloader_keyword


def get_params(model) -> torch.Tensor:
    """
    Returns all the parameters concatenated in a single tensor.
    :return: parameters tensor.
    """
    params = []
    for _, param in model.named_parameters():
        params.append(param.view(-1))
    return torch.cat(params)


def on_task_update(model, big_omega, small_omega, 
                    cached_checkpoint, device, elambda=0):
    """
    Update the regularization after each task learning.
    """
    if big_omega is None:
        big_omega = torch.zeros_like(get_params(model)).to(device)
    # online EWC lambda.
    big_omega += small_omega / ((get_params(model).data - cached_checkpoint) ** 2 + elambda)

    # store parameters checkpoint and reset small_omega
    return big_omega, 0, get_params(model).data.clone().to(device)


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--c", default=20, type=float, help="SI surrogate loss coefficient")
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

    class_list_1 = ["yes", "no", "nine", "three", "bed"]
    class_list_2 = ["up", "down", "wow", "happy", "four"]
    class_list_3 = ["left", "right", "seven", "six", "marvin"]
    class_list_4 = ["on", "off", "house", "zero", "sheila"]
    class_list_5 = ["stop", "go", "dog", "cat", "two"]

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    # initialize and setup Neptune
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet', 'SI'], params=vars(parameters))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build a multi-head setting for learning process.
    total_class_list = []
    learning_tasks = [class_list_1, class_list_2, class_list_3, class_list_4, class_list_5]
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
    else:
        model = None

    # continuous learning by SI.
    big_omega = None
    small_omega = 0
    learned_class_list = []
    trainer = Trainer(parameters, model)
    # store a chaced mdoel checkpoint.
    cached_checkpoint = get_params(trainer.model).data.clone().to(trainer.device)
    for task_id, task_class in enumerate(learning_tasks):
        print(">>>   Learned Class: ", learned_class_list, " To Learn: ", task_class)
        learned_class_list += task_class
        train_loader, test_loader = get_dataloader_keyword(parameters.dpath, task_class, class_encoding, parameters.batch)
        # starting training.
        small_omega = trainer.si_train(task_id, train_loader, test_loader, 
                                        big_omega, small_omega, cached_checkpoint, coefficient=parameters.c, tag=task_id)
        # update the SI parameters.
        big_omega, small_omega, cached_checkpoint = on_task_update(trainer.model, big_omega, 
                                                                    small_omega, cached_checkpoint, trainer.device)
        # start evaluating the CL on previous tasks.
        total_acc = 0
        for val_id, task in enumerate(learning_tasks):
            print(f">>>   Testing on task {val_id}, Keywords: {task}")
            _, val_loader = get_dataloader_keyword(parameters.dpath, task, class_encoding, parameters.batch)
            log_data = Evaluator(trainer.model, tag=f't{task_id}v{val_id}').evaluate(val_loader)
            neptune.log_metric(f'TASK-{task_id}-acc', log_data["test_accuracy"])
            total_acc += log_data["test_accuracy"]
        print(f">>>   Average Accuracy: {total_acc / len(learning_tasks) * 100}")