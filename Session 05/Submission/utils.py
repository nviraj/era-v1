import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


def get_device():
    """
    Function to get the device to be used for training and testing.
    """

    # Check if cuda is available
    cuda = torch.cuda.is_available()

    # Based on check enable cuda if present, if not available
    if cuda:
        final_choice = "cuda"
    else:
        final_choice = "cpu"

    return final_choice, torch.device(final_choice)


def apply_mnist_image_transformations():
    """
    Function to apply the required transformations to the MNIST dataset.
    """
    # Apply the required transformations to the MNIST dataset
    train_transforms = transforms.Compose(
        [
            # Randomly center crop the images to 22x22 and then resize to 28x28
            transforms.RandomApply(
                [
                    transforms.CenterCrop(22),
                ],
                p=0.1,
            ),
            transforms.Resize((28, 28)),
            # Apply random rotation to the images
            transforms.RandomRotation((-15.0, 15.0), fill=0),
            # Convert the images to tensors
            # normalize the images with mean and standard deviation from the whole dataset
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Test data transformations
    test_transforms = transforms.Compose(
        # Convert the images to tensors
        # normalize the images with mean and standard deviation from the whole dataset
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    return train_transforms, test_transforms


def plot_sample_training_images(batch_data, batch_label):
    """
    Function to plot sample images from the training data.
    """

    # Initialize the grid of images and labels
    fig = plt.figure()

    # Display 12 images from the training data
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

    return fig


def get_correct_prediction_count(pPrediction, pLabels):
    """
    Gets the count of correct predictions.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
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
            desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )

    # Append the final loss and accuracy for the epoch
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))


def test(model, device, test_loader, criterion):
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
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data and labels to device
            data, target = data.to(device), target.to(device)

            # Predict using model
            output = model(data)
            # Calculate loss for the batch
            test_loss += criterion(
                output, target
            ).item()  # Remove reduction and fix batch loss

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


def plot_train_test_metrics():
    """Plots the training and test metrics."""

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
