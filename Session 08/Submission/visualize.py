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


# def plot_sample_training_images(batch_data, batch_label, class_label, num_images=30):
#     """Function to plot sample images from the training data."""
#     images, labels = batch_data, batch_label

#     # Calculate the number of images to plot
#     num_images = min(num_images, len(images))
#     # calculate the number of rows and columns to plot
#     num_cols = 5
#     num_rows = int(np.ceil(num_images / num_cols))

#     # Initialize a subplot with the required number of rows and columns
#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

#     # Iterate through the images and plot them in the grid along with class labels

#     for i in range(1, num_images + 1):
#         plt.subplot(num_rows, num_cols, i)
#         plt.tight_layout()
#         plt.axis("off")
#         plt.imshow(images[i - 1])
#         plt.title(class_label[labels[i - 1].item()])
#         plt.xticks([])
#         plt.yticks([])

#     return fig, axs


def convert_back_image(image):
    """Using mean and std deviation convert image back to normal"""
    cifar10_mean = (0.4914, 0.4822, 0.4471)
    cifar10_std = (0.2469, 0.2433, 0.2615)
    image = image.numpy().astype(dtype=np.float32)

    for i in range(image.shape[0]):
        image[i] = (image[i] * cifar10_std[i]) + cifar10_mean[i]

    image = image.clip(0, 1)

    return np.transpose(image, (1, 2, 0))


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

    # Iterate through the images and plot them in the grid along with class labels

    for img_index in range(1, num_images + 1):
        plt.subplot(num_rows, num_cols, img_index)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(convert_back_image(images[img_index - 1]))
        plt.title(class_label[labels[img_index - 1].item()])
        plt.xticks([])
        plt.yticks([])

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
