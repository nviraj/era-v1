"""Module to define the model."""

# Resources
# https://lightning.ai/docs/pytorch/stable/starter/introduction.html
# https://lightning.ai/docs/pytorch/stable/starter/converting.html
# https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from torchmetrics import Accuracy

# What is the start LR and weight decay you'd prefer?
PREFERRED_START_LR = 3e-2
PREFERRED_WEIGHT_DECAY = 1e-5


def detailed_model_summary(model, input_size):
    """Define a function to print the model summary."""

    # https://github.com/TylerYep/torchinfo
    torchinfo.summary(
        model,
        input_size=input_size,
        batch_dim=0,
        col_names=(
            "input_size",
            "kernel_size",
            "output_size",
            "num_params",
            "trainable",
        ),
        verbose=1,
        col_width=16,
    )


############# Assignment 12 Model #############


# This is for Assignment 12
# Model used from Assignment 10 and converted to lightning model
class CustomResNet(pl.LightningModule):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.02

    def __init__(self):
        super().__init__()

        #  Model Notes

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        # 1. Input size: 32x32x3
        self.prep = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer1: X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.layer1_x = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer1: R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.layer1_r1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer 2: Conv 3x3 [256k], MaxPooling2D, BN, ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer 3: X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.layer3_x = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # Layer 3: R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.layer3_r2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        )

        # MaxPooling with Kernel Size 4
        # If stride is None, it is set to kernel_size
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)

        # FC Layer
        self.fc = nn.Linear(512, 10)

        # Define loss function
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        self.loss_function = torch.nn.CrossEntropyLoss()

        # Define accuracy function
        # https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html
        self.accuracy_function = Accuracy(task="multiclass", num_classes=10)

        # Get lr finder values
        self.lr_finder = None

    def print_view(self, x, msg=""):
        """Print shape of the model"""
        if self.print_shape:
            if msg != "":
                print(msg, "\n\t", x.shape, "\n")
            else:
                print(x.shape)

    def forward(self, x):
        """Forward pass"""

        # PrepLayer
        x = self.prep(x)
        self.print_view(x, "PrepLayer")

        # Layer 1
        x = self.layer1_x(x)
        self.print_view(x, "Layer 1, X")
        r1 = self.layer1_r1(x)
        self.print_view(r1, "Layer 1, R1")
        x = x + r1
        self.print_view(x, "Layer 1, X + R1")

        # Layer 2
        x = self.layer2(x)
        self.print_view(x, "Layer 2")

        # Layer 3
        x = self.layer3_x(x)
        self.print_view(x, "Layer 3, X")
        r2 = self.layer3_r2(x)
        self.print_view(r2, "Layer 3, R2")
        x = x + r2
        self.print_view(x, "Layer 3, X + R2")

        # MaxPooling
        x = self.maxpool(x)
        self.print_view(x, "Max Pooling")

        # FC Layer
        # Reshape before FC such that it becomes 1D
        x = x.view(x.shape[0], -1)
        self.print_view(x, "Reshape before FC")
        x = self.fc(x)
        self.print_view(x, "After FC")

        # Softmax
        return F.log_softmax(x, dim=-1)

    def find_optimal_lr(self, train_loader):
        """Use LR Finder to find the best starting learning rate"""

        # https://github.com/davidtvs/pytorch-lr-finder
        # https://github.com/davidtvs/pytorch-lr-finder#notes
        # https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py

        # New optimizer with default LR
        tmp_optimizer = optim.Adam(self.parameters(), lr=PREFERRED_START_LR, weight_decay=PREFERRED_WEIGHT_DECAY)

        # Create LR finder object
        lr_finder = LRFinder(self, optimizer=tmp_optimizer, criterion=self.loss_function)
        lr_finder.range_test(train_loader=train_loader, end_lr=10, num_iter=100)
        # https://github.com/davidtvs/pytorch-lr-finder/issues/88
        _, suggested_lr = lr_finder.plot(suggest_lr=True)
        lr_finder.reset()
        # plot.figure.savefig("LRFinder - Suggested Max LR.png")

        print(f"Suggested Max LR: {suggested_lr}")

        if suggested_lr is None:
            suggested_lr = PREFERRED_START_LR

        return suggested_lr

    # optimiser function
    def configure_optimizers(self):
        """Add ADAM optimizer to the lightning module"""
        optimizer = optim.Adam(self.parameters(), lr=PREFERRED_START_LR, weight_decay=PREFERRED_WEIGHT_DECAY)

        # Percent start for OneCycleLR
        # Handles the case where max_epochs is less than 5
        percent_start = 5 / int(self.trainer.max_epochs)
        if percent_start > 1:
            percent_start = 0.3

        # https://lightning.ai/docs/pytorch/stable/common/optimization.html#total-stepping-batches
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer=optimizer,
                max_lr=self.lr_finder,
                total_steps=int(self.trainer.estimated_stepping_batches),
                pct_start=percent_start,
                div_factor=100,
                three_phase=False,
                anneal_strategy="linear",
                final_div_factor=100,
                verbose=False,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    # Define loss function
    def compute_loss(self, prediction, target):
        """Compute Loss"""

        # Calculate loss
        loss = self.loss_function(prediction, target)

        return loss

    # Define accuracy function
    def compute_accuracy(self, prediction, target):
        """Compute accuracy"""

        # Calculate accuracy
        acc = self.accuracy_function(prediction, target)

        return acc * 100

    # Function to compute loss and accuracy for both training and validation
    def compute_metrics(self, batch):
        """Function to calculate loss and accuracy"""

        # Get data and target from batch
        data, target = batch

        # Generate predictions using model
        pred = self.forward(data)

        # Calculate loss for the batch
        loss = self.compute_loss(prediction=pred, target=target)

        # Calculate accuracy for the batch
        acc = self.compute_accuracy(prediction=pred, target=target)

        return loss, acc

    # training function
    def training_step(self, batch, batch_idx):
        """Training step"""

        # Compute loss and accuracy
        loss, acc = self.compute_metrics(batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, logger=True)
        # Return training loss
        return loss

    # validation function
    def validation_step(self, batch, batch_idx):
        """Validation step"""

        # Compute loss and accuracy
        loss, acc = self.compute_metrics(batch)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, logger=True)
        # Return validation loss
        return loss

    # test function will just use validation step
    def test_step(self, batch, batch_idx):
        """Test step"""

        # Compute loss and accuracy
        loss, acc = self.compute_metrics(batch)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("test_acc", acc, on_epoch=True, logger=True)
        # Return validation loss
        return loss

    def on_train_start(self):
        """Set lr finder value"""
        self.lr_finder = self.find_optimal_lr(train_loader=self.train_dataloader())
