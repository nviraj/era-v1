import matplotlib.pyplot as plt
import torch
from torchvision import datasets


def download_mnist_data(data_path, train_transforms, test_transforms):
    """
    Function to download the MNIST data
    """

    # Download MNIST dataset and apply transformations
    train_data = datasets.MNIST(
        data_path, train=True, download=True, transform=train_transforms
    )
    test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=test_transforms
    )

    return train_data, test_data


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


def plot_sample_training_images(batch_data, num_images=25):
    """
    Function to plot sample images from the training data.
    """

    # Initialize the grid of images and labels
    fig = plt.figure()

    num_images = min(num_images, len(batch_data))

    # Display 12 images from the training data
    for i in range(0, num_images):
        plt.subplot(int(round(num_images / 5, 0)), 5, i + 1)
        # plt.tight_layout()
        plt.axis("off")
        plt.imshow(batch_data[i].squeeze(), cmap="gray_r")
        # plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

    return fig


def get_correct_prediction_count(pPrediction, pLabels):
    """
    Function to get the count of correct predictions.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
