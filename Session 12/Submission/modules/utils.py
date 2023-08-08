"""Module to define utility functions for the project."""
import os

import torch


def get_num_workers(model_run_location):
    """Given a run mode, return the number of workers to be used for data loading."""

    # calculate the number of workers
    num_workers = (os.cpu_count() - 1) if os.cpu_count() > 3 else 2

    # If run_mode is local, use only 2 workers
    num_workers = num_workers if model_run_location == "colab" else 0

    return num_workers


# Function to save the model
# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
def save_model(epoch, model, optimizer, scheduler, batch_size, criterion, file_name):
    """
    Function to save the trained model along with other information to disk.
    """
    # print(f"Saving model from epoch {epoch}...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "batch_size": batch_size,
            "loss": criterion,
        },
        file_name,
    )


# Given a list of train_losses, train_accuracies, test_losses,
# test_accuracies, loop through epoch and print the metrics
def pretty_print_metrics(num_epochs, results):
    """
    Function to print the metrics in a pretty format.
    """
    # Extract train_losses, train_acc, test_losses, test_acc from results
    train_losses = results["train_loss"]
    train_acc = results["train_acc"]
    test_losses = results["test_loss"]
    test_acc = results["test_acc"]

    for i in range(num_epochs):
        print(
            f"Epoch: {i+1:02d}, Train Loss: {train_losses[i]:.4f}, "
            f"Test Loss: {test_losses[i]:.4f}, Train Accuracy: {train_acc[i]:.4f}, "
            f"Test Accuracy: {test_acc[i]:.4f}"
        )


# Given a file path, extract the folder path and create folder recursively if it does not already exist
def create_folder_if_not_exists(file_path):
    """
    Function to create a folder if it does not exist.
    """
    # Extract the folder path
    folder_path = os.path.dirname(file_path)

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
