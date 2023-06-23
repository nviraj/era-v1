import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_correct_prediction_count

############# Train and Test Functions #############


def train_model(
    model, device, train_loader, optimizer, criterion, train_acc, train_losses
):
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
        loss = criterion(pred, target)
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


def test_model(
    model,
    device,
    test_loader,
    criterion,
    test_acc,
    test_losses,
    misclassified_image_data,
):
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
            test_loss += criterion(output, target, reduction="sum").item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1)
            # Check if the prediction is correct
            correct_mask = pred.eq(target)
            # Save the incorrect predictions
            incorrect_indices = ~correct_mask

            # Store images incorrectly predicted, generated predictions and the actual value
            misclassified_image_data["images"].extend(data[incorrect_indices])
            misclassified_image_data["ground_truths"].extend(target[incorrect_indices])
            misclassified_image_data["predicted_vals"].extend(pred[incorrect_indices])

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


############# Assignment 8 Model #############


# This is for Assignment 8
class NormalizationModel(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.05

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
            nn.Conv2d(3, 12, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 12),
        )

        self.C2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(8)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 8),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.C3 = nn.Sequential(
            nn.Conv2d(8, 24, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 24),
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 24),
        )

        self.C5 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 24),
        )

        self.c6 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 12),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.C7 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 32),
        )

        self.C8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 32),
        )

        self.C9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.c10 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=(1, 1), stride=1, bias=False)
        )

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
        x = self.C3(x)
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


############# Assignment 7 Model #############


# This is for Model 01
class Model01(nn.Module):
    """This defines the structure of the NN."""

    def __init__(self):
        super(Model01, self).__init__()

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        # Block 1 - Input Block
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        """Forward pass"""
        x = self.block1(x)
        # print(x.shape)
        x = self.block2(x)
        # print(x.shape)
        x = self.block3(x)
        # print(x.shape)
        x = self.block4(x)
        # print(x.shape)
        x = x.view((x.shape[0], -1))
        # print(x.shape)
        return F.log_softmax(x, dim=-1)


# This is for Model 02
class Model02(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False

    def __init__(self):
        super(Model02, self).__init__()

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        # Model notes
        # Reduced channels in each block compared to Model 01
        # Added max pooling after block 2
        # Added print_shape variable to toggle printing shape of the model

        # Block 1 - Input Block
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=12,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
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
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
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

    def forward(self, x):
        """Forward pass"""
        x = self.block1(x)
        print(x.shape) if self.print_shape else None
        x = self.block2(x)
        print(x.shape) if self.print_shape else None
        x = self.block3(x)
        print(x.shape) if self.print_shape else None
        x = self.block4(x)
        print(x.shape) if self.print_shape else None
        x = self.block5(x)
        print(x.shape) if self.print_shape else None
        x = x.view((x.shape[0], -1))
        print(x.shape) if self.print_shape else None
        return F.log_softmax(x, dim=-1)


# This is for Model 03
class Model03(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False

    def __init__(self):
        super(Model03, self).__init__()

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        #  Model Notes
        # Added padding in 1st block
        # Added batch normalization after every convolution layer
        # Reduced out channels slightly in block 3 to reduce parameters to below 8k
        # Added print_view function to print shape of the model

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


# This is for Model 04
class Model04(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False

    def __init__(self):
        super(Model04, self).__init__()

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        #  Model Notes
        # Added dropouts of 0.1 (Skipped input layer, output layer + penultimate layer and layers with Max Pooling)

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
            nn.Dropout(0.1),
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
            nn.Dropout(0.1),
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
            nn.Dropout(0.1),
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


# This is for Model 05
class Model05(nn.Module):
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
        # Same as model 4, dropout reduced to 0.05
        # Added rotation of 6 degrees to the transforms

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
            nn.Dropout(0.05),
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
            nn.Dropout(0.05),
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
            nn.Dropout(0.05),
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


# This is for Model 06
class Model06(nn.Module):
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
        # Same as model 5, dropout reduced to 0.01
        # Added ReduceLROnPlateau

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


############# Assignment 6 Model #############


# Class to define the NN model for Assignment 6
class Net(nn.Module):
    """This defines the structure of the NN."""

    def __init__(self):
        super(Net, self).__init__()

        # # Reference Links
        # https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
        # https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

        # Using Sequential API to define the model as it seems to be more readable

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Gap layer
        self.block4 = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(kernel_size=5),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        # print(x.shape)

        x = self.block2(x)
        # print(x.shape)

        x = self.block3(x)
        # print(x.shape)

        x = self.block4(x)
        # print(x.shape)

        return F.log_softmax(x)
