"""
Replay the historical data to overcome catastrophic forgetting.

@author huangyz0918
@date 06/08/2021
"""
import time
import torch
import neptune
import argparse
import numpy as np
from model import STFT_TCResnet, MFCC_TCResnet, STFT_MLP, MFCC_RNN, parameter_number
from model import Trainer, Evaluator, get_dataloader_replay

if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
        parser.add_argument("--dpath", default="./dataset", type=str, help="The path of dataset")
        parser.add_argument("--ratio", default=0, type=float, help="Historical data replay ratio")
        parser.add_argument("--log", default=False, action='store_true',
                            help="record the experiment into web neptune.ai")
        parser.add_argument("--tqdm", default=False, action='store_true', help="enable terminal tqdm output.")
        parser.add_argument("--model", default="stft", type=str, help="[stft, mfcc]")
        parser.add_argument("--cha", default=config["tc-resnet8"], type=list,
                            help="The channel of model layers (in list)")
        parser.add_argument("--scale", default=1, type=int, help="The scale of model channel")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="stft", type=str, help="The save name")
        args = parser.parse_args()
        return args


    class_list_0 = ["yes", "no", "nine", 
                    "three", "bed", "up", 
                    "down", "wow", "happy", 
                    "four", "stop", "go",
                    "dog", "cat", "five"]
    class_list_1 = ["tree", "one", "eight"]
    class_list_2 = ["left", "right", "bird"]
    class_list_3 = ["seven", "six", "two"]
    class_list_4 = ["marvin", "on", "sheila"]
    class_list_5 = ["off", "house", "zero"]

    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    # initialize and setup Neptune
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TC-ResNet', 'Rehearsal'],
                              params=vars(parameters))

    # build a multi-head setting for learning process.
    total_class_list = []
    learning_tasks = [class_list_0, class_list_1, class_list_2, class_list_3, class_list_4, class_list_5]
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

    # continuous learning.
    # 50% rehearsal baseline.
    la_list = []
    acc_list = []
    bwt_list = []
    learned_class_list = []
    trainer = Trainer(parameters, model)
    start_time = time.time()
    for task_id, task_class in enumerate(learning_tasks):
        print(">>>   Learned Class: ", learned_class_list, " To Learn: ", task_class)
        learned_class_list += task_class
        if parameters.ratio == 1:
            train_loader, test_loader = get_dataloader_replay(parameters.dpath, learned_class_list, learned_class_list,
                                                              class_encoding,
                                                              replay_ratio=parameters.ratio,
                                                              batch_size=parameters.batch)
        else:
            train_loader, test_loader = get_dataloader_replay(parameters.dpath, task_class, learned_class_list,
                                                              class_encoding,
                                                              replay_ratio=parameters.ratio,
                                                              batch_size=parameters.batch)
        # fine-tune the whole model.
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr, momentum=0.9)
        if parameters.log:
            trainer.model_train(task_id, optimizer, train_loader, test_loader, tag=task_id)
        else:
            trainer.model_train(task_id, optimizer, train_loader, test_loader)
        # the task evaluation.
        total_learned_acc = 0
        for val_id, task in enumerate(learning_tasks):
            print(f">>>   Testing on task {val_id}, Keywords: {task}")
            _, val_loader = get_dataloader_replay(parameters.dpath, task, learned_class_list, class_encoding)
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
        print(f'ACC on TASK {task_id}: {total_learned_acc / (task_id + 1)}')
        if task_id > 0:
            bwt_list.append(np.mean([acc_list[i + 1] - acc_list[i] for i in range(len(acc_list) - 1)]))

    duration = time.time() - start_time
    print(f'Total time {duration}, Avg: {duration / len(learning_tasks)}s')
    print(f'ACC: {np.mean(acc_list)}, std: {np.std(acc_list)}')
    print(f'LA: {np.mean(la_list)}, std: {np.std(la_list)}')
    print(f'BWT: {np.mean(bwt_list)}, std: {np.std(bwt_list)}')
    print(f'Parameter: {parameter_number(trainer.model) / 1024} K')
