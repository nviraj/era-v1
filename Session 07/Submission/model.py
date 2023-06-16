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
    # This defines the structure of the NN.
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


class Model01(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Model01, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )  # output_size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(7, 7),
                padding=0,
                bias=False,
            ),
            # nn.ReLU() NEVER!
        )  # output_size = 1 7x7x10 | 7x7x10x10 | 1x1x10

    def forward(self, x):
        """Forward pass"""
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
