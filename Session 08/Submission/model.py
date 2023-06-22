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
class BNModel(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False

    def __init__(self):
        super().__init__()

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        #  Model Notes

        # Block 1 - Input Block
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),
            nn.Conv2d(
                in_channels=10,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.01),
        )

        # Block 4 - Don't use Max Pooling here
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=14,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.01),
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # Block 5 - Output Block
        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.AdaptiveAvgPool2d(1),
        )

    def print_view(self, x):
        """Print shape of the model"""
        if self.print_shape:
            print(x.shape)

    def forward(self, x):
        """Forward pass"""
        x = self.block1(x)
        self.print_view(x)
        x = self.block2(x)
        self.print_view(x)
        x = self.block3(x)
        self.print_view(x)
        x = self.block4(x)
        self.print_view(x)
        x = self.block5(x)
        self.print_view(x)
        x = x.view((x.shape[0], -1))
        self.print_view(x)
        return F.log_softmax(x, dim=-1)
