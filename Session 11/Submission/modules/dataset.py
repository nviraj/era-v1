"""This file contains functions to download and transform the CIFAR10 dataset"""
# Needed for image transformations
import albumentations as A

# # Needed for padding issues in albumentations
# import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from torchvision import datasets

# Use precomputed values for mean and standard deviation of the dataset
CIFAR_MEAN = (0.4915, 0.4823, 0.4468)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

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


class CIFAR10Transforms(Dataset):
    """Apply albumentations augmentations to CIFAR10 dataset"""

    # Given a dataset and transformations,
    # apply the transformations and return the dataset
    def __init__(self, dataset, transforms):
        self.transforms = transforms
        self.dataset = dataset

    def __getitem__(self, idx):
        # Get the image and label from the dataset
        image, label = self.dataset[idx]

        # Apply transformations on the image
        image = self.transforms(image=np.array(image))["image"]

        return image, label

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return (
            f"CIFAR10Transforms(dataset={self.dataset}, transforms={self.transforms})"
        )

    def __str__(self):
        return (
            f"CIFAR10Transforms(dataset={self.dataset}, transforms={self.transforms})"
        )


def apply_cifar_image_transformations(mean, std, cutout_size):
    """
    Function to apply the required transformations to the MNIST dataset.
    """
    # Apply the required transformations to the MNIST dataset
    train_transforms = A.Compose(
        [
            # normalize the images with mean and standard deviation from the whole dataset
            # https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize
            # # transforms.Normalize(cifar_mean, cifar_std),
            A.Normalize(mean=list(mean), std=list(std)),
            # RandomCrop 32, 32 (after padding of 4)
            # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.PadIfNeeded
            # MinHeight and MinWidth are set to 36 to ensure that the image is padded to 36x36 after padding
            # border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            # Default: cv2.BORDER_REFLECT_101
            A.PadIfNeeded(min_height=36, min_width=36),
            # https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop
            A.RandomCrop(32, 32),
            # CutOut(8, 8)
            # # https://albumentations.ai/docs/api_reference/augmentations/dropout/cutout/#albumentations.augmentations.dropout.cutout.Cutout
            # # Because we normalized the images with mean and standard deviation from the whole dataset, the fill_value is set to the mean of the dataset
            # A.Cutout(
            #     num_holes=1, max_h_size=cutout_size, max_w_size=cutout_size, p=1.0
            # ),
            # https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#coarsedropout-augmentation-augmentationsdropoutcoarse_dropout
            A.CoarseDropout(
                max_holes=1,
                max_height=cutout_size,
                max_width=cutout_size,
                min_holes=1,
                min_height=cutout_size,
                min_width=cutout_size,
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
            A.Normalize(mean=list(mean), std=list(std)),
            # Convert the images to tensors
            ToTensorV2(),
        ]
    )

    return train_transforms, test_transforms


def split_cifar_data(data_path, train_transforms, test_transforms):
    """
    Function to download the MNIST data
    """
    print("Downloading CIFAR10 dataset\n")
    # Download MNIST dataset
    train_data = datasets.CIFAR10(data_path, train=True, download=True)
    test_data = datasets.CIFAR10(data_path, train=False, download=True)

    # Calculate and print the mean and standard deviation of the dataset
    mean, std = calculate_mean_std(train_data)
    print(f"\nMean: {mean}")
    print(f"Std: {std}\n")

    # Apply transforms on the dataset
    # Use the above class to apply transforms on the dataset using albumentations
    train_data = CIFAR10Transforms(train_data, train_transforms)
    test_data = CIFAR10Transforms(test_data, test_transforms)

    print("Transforms applied on the dataset\n")

    return train_data, test_data


def calculate_mean_std(dataset):
    """Function to calculate the mean and standard deviation of CIFAR dataset"""
    data = dataset.data.astype(np.float32) / 255.0
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    return mean, std
