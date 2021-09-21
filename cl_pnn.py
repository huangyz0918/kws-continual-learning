"""
Continuous learning with MLP-PNN.

Reference: Progressive Neural Networks (Google DeepMind)

@author huangyz0918
@date 16/09/2021
"""
import time

import torch
import neptune
import argparse
from model import MLP_PNN
from model import Trainer, Evaluator, get_dataloader_keyword

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
    parser.add_argument("--lc", default=False, action='store_true',
                        help="Test on the task with/without lateral connections")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--batch", default=128, type=int, help="Training batch size")
    parser.add_argument("--step", default=30, type=int, help="Training step size")
    parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
    parser.add_argument("--log", default=False, action='store_true',
                        help="record the experiment into web neptune.ai")
    parser.add_argument("--dpath", default="./dataset", type=str, help="The path of dataset")

    parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
    parser.add_argument("--save", default="stft", type=str, help="The save name")
    parameters = parser.parse_args()

    class_list_1 = ["yes", "no", "nine", "three", "bed",
                    "up", "down", "wow", "happy", "four",
                    "left", "right", "seven", "six", "marvin",
                    "on", "off", "house", "zero", "sheila"]
    class_list_2 = ["stop", "go"]
    class_list_3 = ["dog", "cat"]
    class_list_4 = ["two", "bird"]
    class_list_5 = ["eight", "five"]
    class_list_6 = ["tree", "one"]

    # initialize and setup Neptune
    if parameters.log:
        neptune.init('huangyz0918/kws')
        neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'PNN'], params=vars(parameters))
    class_list = []
    learning_tasks = [class_list_1, class_list_2, class_list_3, class_list_4, class_list_5, class_list_6]

    # initializing the PNN model.
    model = MLP_PNN(256, 129, 129 * 125)
    # start continuous learning.
    model.add_column(len(learning_tasks[0]))  # add the first column for the PNN.
    trainer = Trainer(parameters, model)
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
            trainer.model.add_column(len(task_class), hsize=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr, momentum=0.9)
        # fine-tune the whole model.
        if parameters.log:
            trainer.model_train(task_id, optimizer, train_loader, test_loader, is_pnn=True, tag=f'task{task_id}')
        else:
            trainer.model_train(task_id, optimizer, train_loader, test_loader, is_pnn=True)
        # start evaluating the CL on previous tasks.
        total_acc = 0
        for val_id in range(task_id + 1):
            print(f">>>   Testing on task {val_id}, Keywords: {learning_tasks[val_id]}")
            test_encoding = {category: index for index, category in enumerate(learning_tasks[val_id])}
            _, val_loader = get_dataloader_keyword(parameters.dpath, learning_tasks[val_id], test_encoding,
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
            total_acc += log_data["test_accuracy"]
        print(f">>>   Average Accuracy: {total_acc / (task_id + 1) * 100}")
    duration = time.time() - start_time
    print(f'Training finished, time for {parameters.epoch} epoch: {duration}, average: {duration / parameters.epoch}')
