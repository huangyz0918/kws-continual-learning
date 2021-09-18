"""
Continuous learning with TCResNet-PNN.

Reference: Progressive Neural Networks (Google DeepMind)

@author huangyz0918
@date 16/09/2021
"""

import neptune
import argparse
import torch 
import torch.nn as nn
from model import PNN_Net
from model import Trainer, Evaluator, get_dataloader_keyword


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epoch")
    parser.add_argument("--lc", default=False, action='store_true', help="Test on the task with/without lateral connections")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--batch", default=128, type=int, help="Training batch size")
    parser.add_argument("--step", default=30, type=int, help="Training step size")
    parser.add_argument("--gpu", default=4, type=int, help="Number of GPU device")
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
    neptune.init('huangyz0918/kws')
    neptune.create_experiment(name='kws_model', tags=['pytorch', 'KWS', 'GSC', 'PNN'], params=vars(parameters))

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

    # initializing the PNN model.
    model = PNN_Net(256, 129, 129 * 125)
    # start continuous learning.
    learned_class_list = []
    trainer = Trainer(parameters, model)
    for task_id, task_class in enumerate(learning_tasks):
        learned_class_list += task_class
        # smaller column sizes from 2nd task inwards to limit expansion.
        if task_id > 0:
            trainer.model.add_column(hsize=32)
        else: 
            trainer.model.add_column()

        train_loader, test_loader = get_dataloader_keyword(parameters.dpath, task_class, class_encoding, parameters.batch)
        print(f">>>   Task {task_id}, Testing Keywords: {task_class}")
        # fine-tune the whole model.
        trainer.model_train(task_id, train_loader, test_loader, is_pnn=True, tag=f'task{task_id}')
        # start evaluating the CL on previous tasks.
        total_acc = 0
        for val_id, keyword in enumerate(class_list):
            print(f">>>   Testing on keyword id {val_id}; Keywords: {keyword}")
            _, val_loader = get_dataloader_keyword(parameters.dpath, [keyword], class_encoding, parameters.batch)
            evaluator = Evaluator(trainer.model, tag=f't{task_id}v-{keyword}')
            if parameters.lc:
                log_data = evaluator.pnn_evaluate(val_id, val_loader, with_lateral_con=True)
            else:
                log_data = evaluator.pnn_evaluate(val_id, val_loader)
            neptune.log_metric(f'TASK-{task_id}-keyword-{keyword}-acc', log_data["test_accuracy"])
            total_acc += log_data["test_accuracy"]
        print(f">>>   Average Accuracy: {total_acc / len(class_list) * 100}")