"""This file contains functions to prepare dataloader in the way lightning expects"""
import pytorch_lightning as pl
import torchvision.datasets as datasets
from modules.dataset import CIFAR10Transforms, apply_cifar_image_transformations
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader, random_split


class CIFARDataModule(pl.LightningDataModule):
    """Lightning DataModule for CIFAR10 dataset"""

    def __init__(self, data_path, batch_size, num_workers, seed):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.training_dataset = None
        self.validation_dataset = None
        self.testing_dataset = None
        self.dataloader_dict = {
            "shuffle": True,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "worker_init_fn": self._init_fn,
        }

    def _split_train_val(self, dataset, val_split=0.2):
        """Split the dataset into train and validation sets"""

        # Throw an error if the validation split is not between 0 and 1
        if not 0 < val_split < 1:
            raise ValueError("Validation split must be between 0 and 1")

        # Set seed again, might not be necessary
        seed_everything(int(self.seed))

        # Calculate lengths of each dataset
        train_length = int((1 - val_split) * len(dataset))
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
        seed_everything(int(self.seed))
        # Define the data transformations
        train_transforms, test_transforms = apply_cifar_image_transformations()
        val_transforms = test_transforms

        # Load the appropriate CIFAR10 dataset
        if stage in ["fit", None, "validate"]:
            full_data = datasets.CIFAR10(self.data_path, train=True)
            data_train, data_val = self._split_train_val(full_data, val_split=0.2)

            self.training_dataset = CIFAR10Transforms(data_train, train_transforms)
            self.validation_dataset = CIFAR10Transforms(data_val, val_transforms)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.testing_dataset = CIFAR10Transforms(datasets.CIFAR10(self.data_path, train=False), test_transforms)

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
