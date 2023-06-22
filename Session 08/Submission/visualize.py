import matplotlib.pyplot as plt


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
