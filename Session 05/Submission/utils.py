import matplotlib.pyplot as plt
import torch


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
    Function to get the count of correct predictions.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
