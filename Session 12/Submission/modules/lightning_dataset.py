"""This file contains functions to prepare dataloader in the way lightning expects"""
import pytorch_lightning as pl
import torchvision.datasets as datasets
from lightning_fabric.utilities.seed import seed_everything
from modules.dataset import CIFAR10Transforms, apply_cifar_image_transformations
from torch.utils.data import DataLoader, random_split


class CIFARDataModule(pl.LightningDataModule):
    """Lightning DataModule for CIFAR10 dataset"""

    def __init__(self, data_path, batch_size, seed, val_split=0, num_workers=0):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.seed = seed
        self.val_split = val_split
        self.num_workers = num_workers
        self.dataloader_dict = {
            "shuffle": True,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "worker_init_fn": self._init_fn,
            "persistent_workers": True if self.num_workers > 0 else False,
        }

    def _split_train_val(self, dataset):
        """Split the dataset into train and validation sets"""

        # Throw an error if the validation split is not between 0 and 1
        if not 0 < self.val_split < 1:
            raise ValueError("Validation split must be between 0 and 1")

        # # Set seed again, might not be necessary
        # seed_everything(int(self.seed))

        # Calculate lengths of each dataset
        train_length = int((1 - self.val_split) * len(dataset))
        val_length = len(dataset) - train_length

        # Split the dataset
        train_dataset, val_dataset = random_split(dataset, [train_length, val_length])

        return train_dataset, val_dataset

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#prepare-data
    def prepare_data(self):
        # Download the CIFAR10 dataset if it doesn't exist
        datasets.CIFAR10(self.data_path, train=True, download=True)
        datasets.CIFAR10(self.data_path, train=False, download=True)

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#setup
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.DataHooks.html#lightning.pytorch.core.hooks.DataHooks.setup
    def setup(self, stage=None):
        # seed_everything(int(self.seed))
        # self.prepare_data()
        # Define the data transformations
        train_transforms, test_transforms = apply_cifar_image_transformations()
        val_transforms = test_transforms

        # Load the appropriate CIFAR10 dataset
        full_data = datasets.CIFAR10(self.data_path, train=True)

        # Assign Test split(s) for use in Dataloaders
        data_test = datasets.CIFAR10(self.data_path, train=False)
        self.testing_dataset = CIFAR10Transforms(data_test, test_transforms)

        # Create train and validation datasets
        if self.val_split != 0:
            data_train, data_val = self._split_train_val(full_data)

            self.training_dataset = CIFAR10Transforms(data_train, train_transforms)
            self.validation_dataset = CIFAR10Transforms(data_val, val_transforms)
        else:
            # Only training data here
            self.training_dataset = CIFAR10Transforms(full_data, train_transforms)
            self.validation_dataset = self.testing_dataset

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#train-dataloader
    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            **self.dataloader_dict,
        )

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#val-dataloader
    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            **self.dataloader_dict,
        )

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#test-dataloader
    def test_dataloader(self):
        return DataLoader(
            self.testing_dataset,
            **self.dataloader_dict,
        )

    def _init_fn(self, worker_id):
        seed_everything(int(self.seed) + worker_id)
