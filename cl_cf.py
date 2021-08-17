"""
Example training script of KWS models.

@author huangyz0918
@date 06/08/2021
"""

import neptune
import argparse
from model import TCResNet, STFT_TCResnet, MFCC_TCResnet
from model import Trainer, Evaluator, get_dataloader


if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
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

    class_list_1 = ["yes", "unknown", "silence"]
    class_list_2 = ["down", "unknown", "silence"]
    test_class_list_2 = ["yes", "down", "unknown", "silence"]

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
    # Task 1
    Trainer(parameters, class_list_1, tag='task1', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't1v1').evaluate(test_loader_1) # t1v1 (train on t1 validate on t1)
    # Task 2
    Trainer(parameters, class_list_2, tag='task2', model=model).model_train()
    print(f">>>   Testing Keywords: {class_list_1}")
    Evaluator(model, 't2v1').evaluate(test_loader_1) # t2v1 (train on t2 validate on t1)
    print(f">>>   Testing Keywords: {class_list_2}")
    Evaluator(model, 't2v2').evaluate(test_loader_2) # t2v2 (train on t2 validate on t2)
