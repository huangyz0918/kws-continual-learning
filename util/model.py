import torch


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
        cp.print_color(cp.ColorEnum.YELLOW,
                       "There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        cp.print_color(cp.ColorEnum.YELLOW,
                       "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                           n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids
