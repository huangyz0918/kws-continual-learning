"""
Example training script of KWS models.

@author huangyz0918
@date 06/08/2021
"""

import neptune
import argparse
from model.util.constant import *
from model import TCResNet, STFT_TCResnet, MFCC_TCResnet
from model import Trainer, Evaluator, get_dataloader


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--batch", default=256, type=int, help="Training batch size")
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

    class_list_1 = ["yes", "no", "unknown", "silence"]
    class_list_2 = ["up", "down", "unknown", "silence"]
    class_list_3 = ["left", "right", "unknown", "silence"]
    class_list_4 = ["on", "off", "unknown", "silence"]
    class_list_5 = ["stop", "go", "unknown", "silence"]

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    # initialize and setup Neptune
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model',
                              tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet'],
                              params=vars(parameters))

    if parameters.model == "stft":
        model = STFT_TCResnet(
            filter_length=256, hop_length=129, bins=129,
            channels=parameters.cha, channel_scale=parameters.scale, num_classes=len(class_list_1))
    elif parameters.model == "mfcc":
        model = MFCC_TCResnet(bins=40, channel_scale=parameters.scale, num_classes=len(class_list_1))
    else: 
        model = None

    # load testing dataset
    _, test_loader_1 = get_dataloader(parameters.dpath, class_list_1)
    _, test_loader_2 = get_dataloader(parameters.dpath, class_list_2)
    _, test_loader_3 = get_dataloader(parameters.dpath, class_list_3)
    _, test_loader_4 = get_dataloader(parameters.dpath, class_list_4)
    _, test_loader_5 = get_dataloader(parameters.dpath, class_list_5)
    # Task 1
    Trainer(parameters, class_list_1, cl_mode=CL_REHERSAL, tag='task1', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't1v1').evaluate(test_loader_1) # t1v1 (train on t1 validate on t1)
    # Task 2
    Trainer(parameters, class_list_2, cl_mode=CL_REHERSAL, tag='task2', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't2v1').evaluate(test_loader_1) # t2v1 (train on t2 validate on t1)
    print(f">>>   Testing Keywords: {class_list_2}")
    Evaluator(model, 't2v2').evaluate(test_loader_2) # t2v2 (train on t2 validate on t2)
    # Task 3
    Trainer(parameters, class_list_3, cl_mode=CL_REHERSAL, tag='task3', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't3v1').evaluate(test_loader_1) # t3v1 (train on t3 validate on t1)
    print(f">>>   Testing Keywords: {class_list_2}")
    Evaluator(model, 't3v2').evaluate(test_loader_2) # t3v2 (train on t3 validate on t2)  
    print(f">>>   Testing Keywords: {class_list_3}")
    Evaluator(model, 't3v3').evaluate(test_loader_3) # t3v3 (train on t3 validate on t3)  
    # Task 4
    Trainer(parameters, class_list_4, cl_mode=CL_REHERSAL, tag='task4', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't4v1').evaluate(test_loader_1) # t4v1 (train on t4 validate on t1)
    print(f">>>   Testing Keywords: {class_list_2}")
    Evaluator(model, 't4v2').evaluate(test_loader_2) # t4v2 (train on t4 validate on t2)  
    print(f">>>   Testing Keywords: {class_list_3}")
    Evaluator(model, 't4v3').evaluate(test_loader_3) # t4v3 (train on t4 validate on t3)  
    print(f">>>   Testing Keywords: {class_list_4}")
    Evaluator(model, 't4v4').evaluate(test_loader_4) # t4v4 (train on t4 validate on t4)  
    # Task 5
    Trainer(parameters, class_list_5, cl_mode=CL_REHERSAL, tag='task5', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't5v1').evaluate(test_loader_1) # t5v1 (train on t5 validate on t1)
    print(f">>>   Testing Keywords: {class_list_2}")
    Evaluator(model, 't5v2').evaluate(test_loader_2) # t5v2 (train on t5 validate on t2)  
    print(f">>>   Testing Keywords: {class_list_3}")
    Evaluator(model, 't5v3').evaluate(test_loader_3) # t5v3 (train on t5 validate on t3)  
    print(f">>>   Testing Keywords: {class_list_4}")
    Evaluator(model, 't5v4').evaluate(test_loader_4) # t5v4 (train on t5 validate on t4) 
    print(f">>>   Testing Keywords: {class_list_5}")
    Evaluator(model, 't5v5').evaluate(test_loader_5) # t5v5 (train on t5 validate on t5) 