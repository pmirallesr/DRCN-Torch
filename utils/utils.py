""" A bunch of utilities used throughout the repository """
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import math
import torchvision


def get_dataset(
    dataset_name,
    split,
    data_transforms=[torchvision.transforms.ToTensor()],
    target_transforms=None,
    download=True,
    data_path="data/",
):
    """
    This function accepts the name of a dataset and some settings as a parameter
    and outputs the demanded dataset.
    Args:
        dataset_name: Name of the dataset you want to load. Will throw an
        exception if the name is not within valid_datasets

        split: A string designating the kind of split you want. Currently only
        supports test or train.

        data_transforms: A set of transforms to be applied to the passed images. Should
        at least include torchvision.transforms.toTensor()

        download: Whether you'd like for the data to be downloaded on the local machine

        data_path: Where to download the data, if you want it downloaded

    Returns:
        data_set: The pytorch dataset object designated by dataset_name,
        loaded with the passed parameters.
    Todo:
        * Implement other datasets
        * Handle the case where splits other than train or test are possible
    """

    valid_datasets = ["MNIST", "USPS", "SVHN"]
    if not dataset_name in valid_datasets:
        print("Please enter a valid dataset! {}")
        raise ValueError

    if split == "Train" or split == "train":
        if dataset_name == "MNIST":
            data_set = torchvision.datasets.MNIST(
                data_path + "MNIST/",
                train=True,
                download=download,
                transform=data_transforms,
                target_transform=target_transforms,
            )

        elif dataset_name == "USPS":

            data_set = torchvision.datasets.USPS(
                data_path + "USPS/",
                train=True,
                transform=data_transforms,
                target_transform=target_transforms,
                download=download,
            )
        elif dataset_name == "SVHN":
            data_set = torchvision.datasets.SVHN(
                data_path + "SVHN/",
                split="train",
                download=download,
                transform=data_transforms,
                target_transform=target_transforms,
            )

    elif split == "Test" or split == "test":

        if dataset_name == "MNIST":
            data_set = torchvision.datasets.MNIST(
                data_path + "MNIST/",
                train=False,
                download=download,
                transform=data_transforms,
                target_transform=target_transforms,
            )

        elif dataset_name == "USPS":
            data_set = torchvision.datasets.USPS(
                data_path + "USPS/",
                train=False,
                transform=data_transforms,
                target_transform=target_transforms,
                download=download,
            )
        elif dataset_name == "SVHN":
            data_set = torchvision.datasets.SVHN(
                data_path + "SVHN/",
                split="test",
                download=download,
                transform=data_transforms,
                target_transform=target_transforms,
            )
    else:
        print("Please enter a valid split, train or test!")
        raise ValueError

    return data_set


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


# Some dimension calculation tools

value_error_msg = "Make sure that the attributes of {} are correct"

def calc_conv_output_dimensions(input_shape, conv_layer):
    """
    Calculates the output dimensions from a conv_layer layer with the given parameters
    Args:
        input_shape: The shape of the elements that the layer will be processing
        conv_layer: The pooling layer for whom the output dimensions are needed
    Returns:
        dim: An n-dimensional tuple containing the output dimensions of the conv layer
    """
    assert len(input_shape[2:]) > 0
    variables = [conv_layer.kernel_size, conv_layer.dilation, conv_layer.padding, conv_layer.stride]
    n_dim = len(input_shape[2:])
    # Verify correctness of variables
    for i, elem in enumerate(variables):
        if isinstance(elem, int):
            variables[i] = (elem, elem)
        elif isinstance(elem, tuple):
            if not len(elem) == n_dim:
                print(value_error_msg.format(conv_layer))
                raise ValueError
        else:
            print(value_error_msg.format(conv_layer))
            raise ValueError
    # Init output and calculate
    dim = (input_shape[0], input_shape[1])
    for i in range(n_dim):
        dim_x = math.floor(
            (
                input_shape[i+2]
                + 2 * variables[2][i]
                - variables[1][i] * (variables[0][i] - 1)
                - 1
            )
            / variables[3][i]
            + 1
        )
        # Append to tuple
        dim += (dim_x,)
    return dim


def calc_pool_output_dimensions(input_shape, pool_layer):
    """
    Calculates the output dimensions from a pool_layer layer with the given parameters
    Args:
        input_shape: The shape of the elements that the layer will be processing
        pool_layer: The pool_layer layer for whom the output dimensions are needed
    """
    assert len(input_shape[2:]) > 0
    variables = [pool_layer.kernel_size, pool_layer.dilation, pool_layer.padding, pool_layer.stride]
    n_dim = len(input_shape[2:])
    # Verify correctness of variables
    for i, elem in enumerate(variables):
        if isinstance(elem, int):
            variables[i] = (elem, elem)
        elif isinstance(elem, tuple):
            if not len(elem) == ndim:
                print(value_error_msg.format(pool_layer))
                raise ValueError
        else:
            print(value_error_msg.format(pool_layer))
            raise ValueError
    # Init output and calculate
    dim = (input_shape[0], input_shape[1])
    for i in range(n_dim):
        dim_x = ((input_shape[i+2] + 2 * variables[2][i] - variables[1][i] * (variables[0][i] - 1) - 1) / variables[3][i]) + 1
        dim += (dim_x,)
    return dim


# Some padding calculation tools.


def calc_conv_same_padding(input_shape, conv_net, n_dim=2):
    """
    Calculates padding required for conv_net to produce output of same dimensions as input_shape
    Args:
       input_shape: an example of the data that will be processed by the pooling layer, in torchivision.tensor form
       conv_net: The layer for whom the padding is needed, with initialized stride, kernel, and dilation
       n_dim: To Be Implemented for layers processing input_shape other than 2-dimensional
    """
    pad_h = math.ceil(
        (
            conv_net.kernel_size[0]
            - input_shape[2] * (1 - conv_net.stride[0])
            - conv_net.stride[0]
        )
        / 2
    )
    pad_w = math.ceil(
        (
            conv_net.kernel_size[1]
            - input_shape[3] * (1 - conv_net.stride[1])
            - conv_net.stride[1]
        )
        / 2
    )
    return pad_h, pad_w


def calc_pool_same_padding(input_shape, pool_net, n_dim=2):
    """
    Calculates padding required for pool_net to produce output of same dimensions as input_shape
    Args:
        input_shape: an example of the data that will be processed by the pooling layer, in torchivision.tensor form
        pool_net: The layer for whom the padding is needed, with initialized stride, kernel, and dilation
        n_dim: To Be Implemented for layers processing input_shape other than 2-dimensional
    """
    pad_h = math.ceil(
        (
            (input_shape[2] - 1) * pool_net.stride[0]
            + 1
            + pool_net.dilation * (pool_net.kernel_size[0] - 1)
            - input_shape.shape[2]
        )
        / 2
    )
    pad_w = math.ceil(
        (
            (input_shape[3] - 1) * pool_net.stride[1]
            + 1
            + pool_net.dilation * (pool_net.kernel_size[1] - 1)
            - input_shape[3]
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
