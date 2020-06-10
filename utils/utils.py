""" A bunch of utilities used throughout the repository """
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import math

# Fix the random seeds
def set_seeds(random_seed=random.randint(1, 100000)):
    """
    Based on:
    http://tiny.cc/22umqz
    Args
        random_seed: The number to which the random seeds must be set
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    os.environ["PYTHONASHSEED"] = str(random_seed)
    random.seed(random_seed)


# functions to show an image
def imshow(img):
    """
    Shows the given image
    Args:
        img: an image in torchvision.tensor form
    """
    npimg = img.numpy()
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Some padding calculation tools.


def calc_conv_same_padding(input, conv_net, n_dim=2):
    """
    Calculates padding required for conv_net to produce output of same dimensions as input
    Args:
       input: an example of the data that will be processed by the pooling layer, in torchivision.tensor form
       conv_net: The layer for whom the padding is needed, with initialized stride, kernel, and dilation
       n_dim: To Be Implemented for layers processing input other than 2-dimensional
    """
    pad_h = math.ceil(
        (
            conv_net.kernel_size[0]
            - input.shape[2] * (1 - conv_net.stride[0])
            - conv_net.stride[0]
        )
        / 2
    )
    pad_w = math.ceil(
        (
            conv_net.kernel_size[1]
            - input.shape[3] * (1 - conv_net.stride[1])
            - conv_net.stride[1]
        )
        / 2
    )
    return pad_h, pad_w


def calc_pool_same_padding(input, pool_net, n_dim=2):
    """
    Calculates padding required for pool_net to produce output of same dimensions as input
    Args:
        input: an example of the data that will be processed by the pooling layer, in torchivision.tensor form
        pool_net: The layer for whom the padding is needed, with initialized stride, kernel, and dilation
        n_dim: To Be Implemented for layers processing input other than 2-dimensional
    """
    pad_h = math.ceil(
        (
            (input.shape[2] - 1) * pool_net.stride[0]
            + 1
            + pool_net.dilation * (pool_net.kernel_size[0] - 1)
            - input.shape[2]
        )
        / 2
    )
    pad_w = math.ceil(
        (
            (input.shape[3] - 1) * pool_net.stride[1]
            + 1
            + pool_net.dilation * (pool_net.kernel_size[1] - 1)
            - input.shape[3]
        )
        / 2
    )
    return pad_h, pad_h, pad_w, pad_w


# Conv2d layer weight init


def weights_init(m):
    """
    Initialize the given layer with a xavier_uniform distribution for its input
    weights and a uniform 0.1 for its biases
    Args:
        m: A pytorch model that is an instance of nn.Conv2d
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        nn.init.constant_(m.bias.data, 0.1)


# Module evaluation


def get_labelling_accuracy(data_loader, model):
    """
    Args:
        data_loader: a pytorch Data_Loader object with the dataset the model must be evaluated against
        model: a pytorch model inheriting from nn.Module
    """
    correct = 0
    total = 0
    if next(model.parameters()).is_cuda:
        device = "cuda"
    else:
        device = "cpu"
    for batch_id, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        forward = model(data)
        pred = torch.max(forward, 1).indices
        total += target.size(0)
        correct += (pred == target).sum().item()
    return correct, total, correct / total * 100


# Experiment management


def calc_experiments_amount(experiment_ranges):
    """
    Returns the total experiments implied by the given experiment ranges. Total = product over all keys
    of len(experiment_ranges[key])
    Args:
        experiment_ranges: A dictionary where keys = experiment variables and
        values = values of those variables for a set of experiments
    """
    experiment_amount = 1
    for parameter_set in experiment_ranges.values():
        experiment_amount *= len(parameter_set)
    return experiment_amount


def create_exp_name(experiment_ranges):
    """
    Returns a string of the form "Var1_Var2_Var3" that describes the set of experiments
    Args:
        experiment_ranges: A dictionary where keys = experiment variables and
        values = values of those variables for a set of experiments
    """
    exp_name = ""
    for variable in experiment_ranges.keys():
        if len(experiment_ranges[variable]) > 1:
            exp_name += variable + "_"
    return exp_name[:-1]
