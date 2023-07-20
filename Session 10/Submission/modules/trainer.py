"""Module to define the train and test functions."""

# from functools import partial

import torch
from tqdm import tqdm
from utils import get_correct_prediction_count, save_model

# # # Reset tqdm
# # tqdm._instances.clear()
# tqdm = partial(tqdm, position=0, leave=True)

############# Train and Test Functions #############


def train_model(model, device, train_loader, optimizer, criterion):
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
        msg = f"Train: Loss={loss.item():0.4f}, Batch_id={batch_idx}, Accuracy={100*correct/processed:0.2f}"
        pbar.set_description(desc=msg)

    # Close the progress bar
    pbar.close()

    # Return the final loss and accuracy for the epoch
    current_train_accuracy = 100 * correct / processed
    current_train_loss = train_loss / len(train_loader)

    return current_train_accuracy, current_train_loss


def test_model(
    model,
    device,
    test_loader,
    criterion,
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

    # Return the final loss and accuracy for the epoch
    current_test_accuracy = 100.0 * correct / len(test_loader.dataset)
    current_test_loss = test_loss

    # Print the final test loss and accuracy
    print(
        f"Test set: Average loss: {current_test_loss:.4f}, ",
        f"Accuracy: {correct}/{len(test_loader.dataset)} ",
        "({current_test_accuracy:.2f}%)",
    )

    # Return the final loss and accuracy for the epoch
    return current_test_accuracy, current_test_loss
