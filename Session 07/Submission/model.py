import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_correct_prediction_count

# test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


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


def plot_train_test_metrics(train_losses, train_acc, test_losses, test_acc):
    """
    Function to plot the training and test metrics.
    """

    # Plot the graphs in a 2x2 grid showing the training and test metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

    return fig, axs


# Define the neural network architecture
# This is for Model 00
class Model00(nn.Module):
    """This defines the structure of the NN"""

    def __init__(self):
        super(Model00, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Defines a forward function for our neural network
        """
        x = F.relu(self.conv1(x), 2)  # 28>26 | 1>3 | 1>1
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 26>24>12 | 3>5>6 | 1>1>2
        # print(x.shape)
        x = F.relu(self.conv3(x), 2)  # 12>10 | 6>10 | 2>2
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))  # 10>8>4 | 10>14>16 | 2>2>4
        # print(x.shape)
        x = x.view(-1, 4096)  # 4*4*256 = 4096
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)

        return F.log_softmax(x, dim=1)


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
        # Added dropouts (Skipped input layer, output layer + penultimate layer and layers with Max Pooling)

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
