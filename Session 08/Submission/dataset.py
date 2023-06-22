import numpy as np
from torchvision import datasets


def split_cifar_data(data_path, train_transforms, test_transforms):
    """
    Function to download the MNIST data
    """

    # Download MNIST dataset and apply transformations
    train_data = datasets.CIFAR10(
        data_path, train=True, download=True, transform=train_transforms
    )
    test_data = datasets.CIFAR10(
        data_path, train=False, download=True, transform=test_transforms
    )

    return train_data, test_data


def calculate_mean_std(dataset):
    """Function to calculate the mean and standard deviation of CIFAR dataset"""
    data = dataset.data.astype(np.float32) / 255.0
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    return mean, std
