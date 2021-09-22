"""
@author huangyz0918
@date 06/08/2021
"""

import time
import quadprog
import numpy as np

import torch
import torch.nn as nn


class ColorEnum:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def print_color(color, *msgs):
    print(color, *msgs, ColorEnum.END)


def parameter_number(model):
    num_params = 0
    for tensor in list(model.parameters()):
        tensor = tensor.view(-1)
        num_params += len(tensor)
    return num_params


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print_color(ColorEnum.YELLOW, "There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print_color(ColorEnum.YELLOW,
                    "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                        n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def readlines(datapath):
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines


def sample_dataset(dataloader):
    sample = 0
    start = time.time()
    for index, data in enumerate(dataloader):
        sample = data
        if index == 0:
            break
    print("batch sampling time:  ", time.time() - start)
    return sample


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params:
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params:
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


def get_params(model: nn.Module) -> torch.Tensor:
    """
    Returns all the parameters concatenated in a single tensor.
    :return: parameters tensor.
    """
    params = []
    for _, param in model.named_parameters():
        params.append(param.view(-1))
    return torch.cat(params)


def set_params(model, new_params: torch.Tensor) -> None:
    """
    Sets the parameters to a given value.
    :param new_params: concatenated values to be set (??)
    """
    assert new_params.size() == get_params(model).size()
    progress = 0
    for _, param in model.named_parameters():
        cand_params = new_params[progress: progress + torch.tensor(param.size()).prod()].view(param.size())
        progress += torch.tensor(param.size()).prod()
        param.data = cand_params


def get_gards(model: nn.Module) -> torch.Tensor:
    """
    Returns all the gardians concatenated in a single tensor.
    :return: gardians tensor.
    """
    grads = []
    for _, param in model.named_parameters():
        grads.append(param.grad.view(-1))
    return torch.cat(grads)


def get_grad_dim(model: nn.Module) -> torch.Tensor:
    grad_dims = []
    for _, param in model.named_parameters():
        grad_dims.append(param.data.numel())
    return grad_dims
