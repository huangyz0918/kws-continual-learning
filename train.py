"""
Example training script of KWS models.

@author huangyz0918
@date 06/08/2021
"""
import torch
import neptune
import argparse
from model import TCResNet, STFT_TCResnet, MFCC_TCResnet, STFT_MLP, MFCC_RNN
from model import Trainer, get_dataloader_keyword

if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
        parser.add_argument("--tqdm", default=False, action='store_true', help="enable terminal tqdm output.")
        parser.add_argument("--dpath", default="./dataset", type=str, help="The path of dataset")

        parser.add_argument("--model", default="stft", type=str, help=["stft", "mfcc"])
        parser.add_argument("--cha", default=config["tc-resnet8"], type=list,
                            help="The channel of model layers (in list)")
        parser.add_argument("--scale", default=3, type=int, help="The scale of model channel")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="stft", type=str, help="The save name")
        args = parser.parse_args()
        return args


    class_list = ["yes", "no", "unknown", "silence"]

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    # initialize and setup Neptune
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet'], params=vars(parameters))

    if parameters.model == "stft":
        model = STFT_TCResnet(
            filter_length=256, hop_length=129, bins=129,
            channels=parameters.cha, channel_scale=parameters.scale, num_classes=len(class_list))
    elif parameters.model == "mfcc":
        model = MFCC_TCResnet(bins=40, channel_scale=parameters.scale, num_classes=len(class_list))
    elif parameters.model == "stft-mlp":
        model = STFT_MLP(filter_length=256, hop_length=129, bins=129, num_classes=len(class_list))
    elif parameters.model == "mfcc-rnn":
        model = MFCC_RNN(n_mfcc=40, sampling_rate=16000, num_classes=len(class_list),
                         n_layers=3)  # sample length for the dataset is 16000.
    else:
        model = None

    optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr, momentum=0.9)
    class_encoding = {category: index for index, category in enumerate(class_list)}
    train_loader, test_loader = get_dataloader_keyword(parameters.dpath, class_list, class_encoding, parameters.batch)
    Trainer(parameters, model).model_train(0, optimizer, train_loader, test_loader, tag='CF')  # task id: 0
