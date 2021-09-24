"""
Evaluator class for the PyTorch model.

@author huangyz0918
@date 15/08/2021
"""
import torch
import neptune
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, class_list=None, tag=None):
        self.tag = tag
        self.class_list = class_list
        self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.log_data = {"test_accuracy": 0.0, "test_total": 0, "test_correct": 0}

    def evaluate(self, data_loader):
        self.model.eval()
        for batch_idx, (waveform, labels) in enumerate(data_loader):
            with torch.no_grad():
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                logits = self.model(waveform)

                _, predict = torch.max(logits.data, 1)
                self.log_data["test_total"] += labels.size(0)
                self.log_data["test_correct"] += (predict == labels).sum().item()
                self.log_data["test_accuracy"] = self.log_data["test_correct"] / self.log_data["test_total"]
        if self.tag:
            neptune.log_metric(f'{self.tag}-test_accuracy', self.log_data["test_accuracy"])
        if self.class_list:
            print(f'>>>   Test on {self.class_list}, Acc: {100 * self.log_data["test_accuracy"]}')
        else:
            print(f'>>>   Test Acc: {100 * self.log_data["test_accuracy"]}')

        return self.log_data

    def pnn_evaluate(self, task_id, data_loader, with_lateral_con=False):
        """
        with_lateral_con: test on the task with/without lateral connections.
        """
        if with_lateral_con:
            l_w = [1] * task_id
        else:
            l_w = [0] * task_id

        self.model.eval()
        for batch_idx, (waveform, labels) in enumerate(data_loader):
            with torch.no_grad():
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                logits = self.model(waveform, task_id, lateral_weights=l_w)

                _, predict = torch.max(logits.data, 1)
                self.log_data["test_total"] += labels.size(0)
                self.log_data["test_correct"] += (predict == labels).sum().item()
                self.log_data["test_accuracy"] = self.log_data["test_correct"] / self.log_data["test_total"]

        if self.tag:
            neptune.log_metric(f'{self.tag}-test_accuracy', self.log_data["test_accuracy"])
        if self.class_list:
            print(f'>>>   Test on {self.class_list}, Acc: {100 * self.log_data["test_accuracy"]}')
        else:
            print(f'>>>   Test Acc: {100 * self.log_data["test_accuracy"]}')

        return self.log_data
