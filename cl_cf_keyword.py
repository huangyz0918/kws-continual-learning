"""
Continuous learning with basic finetuning.

@author huangyz0918
@date 06/08/2021
"""

import neptune
import argparse
from model.util.constant import *
from model import TCResNet, STFT_TCResnet, MFCC_TCResnet
from model import Trainer, Evaluator, get_dataloader_keyword


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
    neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet', 'Keyword'], params=vars(parameters))

    if parameters.model == "stft":
        model = STFT_TCResnet(
            filter_length=256, hop_length=129, bins=129,
            channels=parameters.cha, channel_scale=parameters.scale, num_classes=len(class_list_1))
    elif parameters.model == "mfcc":
        model = MFCC_TCResnet(bins=40, channel_scale=parameters.scale, num_classes=len(class_list_1))
    else: 
        model = None

    learning_tasks = [class_list_1, class_list_2, class_list_3, class_list_4, class_list_5]
    for task_id, task_class in enumerate(learning_tasks):
        train_loader, test_loader = get_dataloader_keyword(parameters.dpath, task_class, parameters.batch)
        model = Trainer(parameters, task_class, train_loader, test_loader,
                        cl_mode=CL_NONE, tag=f'task{task_id}', model=model).model_train()
        print(f">>>   Task {task_id}, Testing Keywords: {task_class}")
        for val_id in range(task_id + 1):
            _, test_loader = get_dataloader_keyword(parameters.dpath, learning_tasks[val_id], parameters.batch)
            Evaluator(model, f't{task_id}v{val_id}').evaluate(test_loader) 