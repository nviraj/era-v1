import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_correct_prediction_count


def train_model(model, device, train_loader, optimizer, train_acc, train_losses):
    """
    Function to train the model on the train dataset.
    """

    # Initialize the model to train mode
    model.train()
    # Initialize progress bar
    pbar = tqdm(train_loader)

    # Reset the loss and correct predictions for the epoch
    train_loss = 0
    correct = 0
    processed = 0

    # Iterate over the train loader
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data and labels to device
        data, target = data.to(device), target.to(device)
        # Clear the gradients for the optimizer to avoid accumulation
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss for the batch
        loss = F.nll_loss(pred, target)
        # Update the loss
        train_loss += loss.item()

        # Backpropagation to calculate the gradients
        loss.backward()
        # Update the weights
        optimizer.step()

        # Get the count of correct predictions
        correct += get_correct_prediction_count(pred, target)
        processed += len(data)

        # Update the progress bar
        pbar.set_description(
            desc=f"Train: Loss={loss.item():0.4f}, Batch_id={batch_idx}, Accuracy={100*correct/processed:0.2f}"
        )

    # Append the final loss and accuracy for the epoch
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test_model(model, device, test_loader, test_acc, test_losses):
    """
    Function to test the model on the test dataset.
    """

    # Initialize the model to evaluation mode
    model.eval()

    # Reset the loss and correct predictions for the epoch
    test_loss = 0
    correct = 0

    # Disable gradient calculation while testing
    with torch.no_grad():
        for data, target in test_loader:
            # Move data and labels to device
            data, target = data.to(device), target.to(device)

            # Predict using model
            output = model(data)
            # Calculate loss for the batch
            test_loss += F.nll_loss(output, target, reduction="sum").item()

            # Get the count of correct predictions
            correct += get_correct_prediction_count(output, target)

    # Calculate the final loss
    test_loss /= len(test_loader.dataset)
    # Append the final loss and accuracy for the epoch
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print the final test loss and accuracy
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)"
    )


# This is for Model 06
class NormalizationModel(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.025

    def __init__(self, normalization_method="batch", num_groups=2):
        super().__init__()

        self.norm = normalization_method
        self.num_group = num_groups

        # Throw an error if normalization method does not match list of acceptable values
        allowed_norm = ["batch", "layer", "group"]
        if self.norm not in allowed_norm:
            raise ValueError(f"Normalization method must be one of {allowed_norm}")

        # if layer normalisation is chosen change num_groups to 1
        if self.norm == "layer":
            self.num_group = 1

        # if group normalisation is chosen throw an error if num_groups is less than 2
        if self.norm == "group":
            if self.num_group < 2:
                raise ValueError(
                    "Number of groups must be greater than 1 for group normalization"
                )

        # General Notes

        # ReLU used after every Convolution layer
        # Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        #  Model Notes

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 12),
        )

        self.C2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.C3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.C5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.c6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.C7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.C8 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.C9 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.c10 = nn.Sequential(nn.Conv2d(16, 10, kernel_size=1, stride=1, bias=False))

    def print_view(self, x):
        """Print shape of the model"""
        if self.print_shape:
            print(x.shape)

    def forward(self, x):
        """Forward pass"""

        x = self.C1(x)
        self.print_view(x)
        x = self.C2(x)
        self.print_view(x)
        x = self.c3(x)
        self.print_view(x)
        x = self.pool1(x)
        self.print_view(x)
        x = self.C4(x)
        self.print_view(x)
        x = self.C4(x)
        self.print_view(x)
        x = self.C5(x)
        self.print_view(x)
        x = self.c6(x)
        self.print_view(x)
        x = self.pool2(x)
        self.print_view(x)
        x = self.C7(x)
        self.print_view(x)
        x = self.C8(x)
        self.print_view(x)
        x = self.C9(x)
        self.print_view(x)
        x = self.gap(x)
        self.print_view(x)
        x = self.c10(x)
        self.print_view(x)
        x = x.view((x.shape[0], -1))
        self.print_view(x)
        x = F.log_softmax(x, dim=1)

        return x
