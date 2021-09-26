"""
Continuous learning with TC-PNN.

Reference: Progressive Neural Networks (Google DeepMind)

@author huangyz0918
@date 16/09/2021
"""
import time

import torch
import neptune
import argparse
import numpy as np
from model import TC_PNN, parameter_number
from model import Trainer, Evaluator, get_dataloader_keyword

output_results = {}

if __name__ == "__main__":
    def options(config):
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=1, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--lc", default=False, action='store_true',
                            help="Test on the task with/without lateral connections")
        parser.add_argument("--tqdm", default=False, action='store_true', help="enable terminal tqdm output.")
        parser.add_argument("--log", default=False, action='store_true', help="record the experiment into web neptune.ai")
        parser.add_argument("--ek", default=False, action='store_true', help="evaluate the CL by keywords")
        parser.add_argument("--c", default=5, type=float, help="SI surrogate loss coefficient")
        parser.add_argument("--batch", default=128, type=int, help="Training batch size")
        parser.add_argument("--step", default=30, type=int, help="Training step size")
        parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
        parser.add_argument("--dpath", default="./dataset", type=str, help="The path of dataset")

        parser.add_argument("--cha", default=config["tc-resnet8"], type=list,
                            help="The channel of model layers (in list)")
        parser.add_argument("--scale", default=1, type=float, help="The scale of model channel")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="stft", type=str, help="The save name")
        args = parser.parse_args()
        return args


    config = {
        "tc-resnet8": [16, 24, 32, 48],
        "tc-resnet14": [16, 24, 24, 32, 32, 48, 48]}

    parameters = options(config)

    c1 = [16, 24, 32, 48]
    c2 = [16, 48]

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

    # initialize and setup Neptune
    if parameters.log:
        neptune.init('huangyz0918/kws')
        neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'TCPNN'], params=vars(parameters))
    class_list = []
    learning_tasks = [class_list_0, class_list_1, class_list_2, class_list_3, class_list_4, class_list_5]

    # initializing the TC-PNN model.
    model = TC_PNN(bins=129, filter_length=256, hop_length=129)
    # start continuous learning.
    model.add_column(len(learning_tasks[0]), c1, parameters.scale)  # add the first column for the PNN.
    trainer = Trainer(parameters, model)
    acc_list = []
    bwt_list = []
    la_list = []
    learned_class_list = []
    start_time = time.time()
    for task_id, task_class in enumerate(learning_tasks):
        print(">>>   Learned Class: ", learned_class_list, " To Learn: ", task_class)
        learned_class_list += task_class
        class_encoding = {category: index for index, category in enumerate(task_class)}
        train_loader, test_loader = get_dataloader_keyword(parameters.dpath, task_class, class_encoding,
                                                           parameters.batch)
        # smaller column sizes from 2nd task inwards to limit expansion.
        if task_id > 0:
            trainer.model.add_column(len(task_class), c2, parameters.scale)
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr, momentum=0.9)
        # fine-tune the whole model.
        if parameters.log:
            trainer.model_train(task_id, optimizer, train_loader, test_loader, is_pnn=True, tag=f'task{task_id}')
        else:
            trainer.model_train(task_id, optimizer, train_loader, test_loader, is_pnn=True)
        print(f'Parameter on TASK {task_id}: {parameter_number(trainer.model) / 1024} K')
        # start evaluating the CL on previous tasks.
        total_learned_acc = 0
        if parameters.ek:
            evaluate_list = class_list
        else: 
            evaluate_list = learning_tasks
        for val_id in range(task_id + 1):
            print(f">>>   Testing on task {val_id}, Keywords: {evaluate_list[val_id]}")
            test_encoding = {category: index for index, category in enumerate(evaluate_list[val_id])}
            _, val_loader = get_dataloader_keyword(parameters.dpath, evaluate_list[val_id], test_encoding,
                                                   parameters.batch)
            if parameters.log:
                evaluator = Evaluator(trainer.model, tag=f't{task_id}v{val_id}')
            else:
                evaluator = Evaluator(trainer.model)
            if parameters.lc:
                log_data = evaluator.pnn_evaluate(val_id, val_loader, with_lateral_con=True)
            else:
                log_data = evaluator.pnn_evaluate(val_id, val_loader)
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
