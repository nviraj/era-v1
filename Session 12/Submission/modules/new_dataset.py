"""This file contains functions to download and transform the CIFAR10 dataset"""
# Needed for image transformations
import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, random_split

# Use precomputed values for mean and standard deviation of the dataset
CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CUTOUT_SIZE = 16

# Create class labels and convert to tuple
CIFAR_CLASSES = tuple(
    c.capitalize()
    for c in [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
)


class CIFARDataModule(pl.LightningDataModule):
    mean = CIFAR_MEAN
    std = CIFAR_STD
    num_classes = 10
    classes = CIFAR_CLASSES
    image_height = 32
    image_width = 32
    padding = 4
    cutout_size = 8

    def __init__(self, data_path, batch_size, num_workers, seed, self.cutout_size=16):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.self.cutout_size = self.cutout_size
        self.cifar_train = None
        self.cifar_val = None
        self.cifar_test = None

    def prepare_data(self):
        # Download the CIFAR10 dataset if it doesn't exist
        datasets.CIFAR10(self.data_path, train=True, download=True)
        datasets.CIFAR10(self.data_path, train=False, download=True)

    def setup(self, stage=None):
        # Define the data transformations
        train_transforms, test_transforms = self.apply_cifar_image_transformations()

        # Load the CIFAR10 dataset
        cifar_full = datasets.CIFAR10(self.data_path, train=True, transform=train_transforms)
        self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
        self.cifar_test = datasets.CIFAR10(self.data_path, train=False, transform=test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def apply_cifar_image_transformations(self):
        # Apply the required transformations to the MNIST dataset
        train_transforms = A.Compose(
            [
                # normalize the images with mean and standard deviation from the whole dataset
                # https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize
                # # transforms.Normalize(cifar_mean, cifar_std),
                A.Normalize(mean=list(self.mean), std=list(self.std)),
                # RandomCrop 32, 32 (after padding of 4)
                # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
                # MinHeight and MinWidth are set to 36 to ensure that the image is padded to 36x36 after padding
                A.PadIfNeeded(min_height=self.image_height + self.padding, min_width=self.image_width + self.padding),
                # https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop
                A.RandomCrop(self.image_height, self.image_width),
                # https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#coarsedropout-augmentation-augmentationsdropoutcoarse_dropout
                A.CoarseDropout(
                    max_holes=1,
                    max_height=self.cutout_size,
                    max_width=self.cutout_size,
                    min_holes=1,
                    min_height=self.cutout_size,
                    min_width=self.cutout_size,
                    p=1.0,
                ),
                # Convert the images to tensors
                # # transforms.ToTensor(),
                ToTensorV2(),
            ]
        )

        # Test data transformations
        test_transforms = A.Compose(
            # Convert the images to tensors
            # normalize the images with mean and standard deviation from the whole dataset
            [
                A.Normalize(mean=list(self.mean), std=list(self.std)),
                # Convert the images to tensors
                ToTensorV2(),
            ]
        )

        return train_transforms, test_transforms