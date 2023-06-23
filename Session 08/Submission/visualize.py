import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

# def plot_sample_training_images(batch_data, num_images=25):
#     """
#     Function to plot sample images from the training data.
#     """

#     # Initialize the grid of images and labels
#     fig = plt.figure()

#     num_images = min(num_images, len(batch_data))

#     # Display 12 images from the training data
#     for i in range(0, num_images):
#         plt.subplot(int(round(num_images / 5, 0)), 5, i + 1)
#         # plt.tight_layout()
#         plt.axis("off")
#         plt.imshow(batch_data[i].squeeze(), cmap="gray_r")
#         # plt.title(batch_label[i].item())
#         plt.xticks([])
#         plt.yticks([])

#     return fig


def plot_sample_training_images(batch_data, batch_label, class_label, num_images=30):
    """Function to plot sample images from the training data."""
    images, labels = batch_data, batch_label

    # Calculate the number of images to plot
    num_images = min(num_images, len(images))
    # calculate the number of rows and columns to plot
    num_cols = 5
    num_rows = int(np.ceil(num_images / num_cols))

    # Initialize a subplot with the required number of rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_images:
                axs[i, j].axis("off")
                tmp_img = images[index]

                image = np.transpose(
                    images[index], (1, 2, 0)
                )  # Transpose image dimensions if needed
                axs[i, j].imshow(image)
                axs[i, j].set_title(class_label[labels[index].item()])

    plt.tight_layout()

    return fig, axs


# def plot_sample_training_images(batch_data, batch_label, num_images=30):
#     """
#     Show images in a grid with 5 columns.

#     Args:
#         batch_data (Tensor): Batch of images.
#         labels (Tensor): Batch of labels.
#         num_images (int): Number of images to show.
#     """

#     # https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html
#     grid_img = torchvision.utils.make_grid(batch_data[:num_images], nrow=5)
#     print("Labels:", batch_label[:num_images])
#     torchvision.utils.imshow(grid_img)


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
