"""Module to define the train and test functions."""

# from functools import partial

import modules.config as config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler

# Define all the required pytorch lightning callbacks
lr_monitor = LearningRateMonitor(logging_interval="step")


############# Train and Test Functions #############


# def train_model(model, device, train_loader, optimizer, criterion):
#     """
#     Function to train the model on the train dataset.
#     """

#     # Initialize the model to train mode
#     model.train()

#     # Initialize progress bar
#     pbar = tqdm(train_loader)

#     # Reset the loss and correct predictions for the epoch
#     train_loss = 0
#     correct = 0
#     processed = 0

#     # Iterate over the train loader
#     for batch_idx, (data, target) in enumerate(pbar):
#         # Move data and labels to device
#         data, target = data.to(device), target.to(device)
#         # Clear the gradients for the optimizer to avoid accumulation
#         optimizer.zero_grad()

#         # Predict
#         pred = model(data)

#         # Calculate loss for the batch
#         loss = criterion(pred, target)
#         # Update the loss
#         train_loss += loss.item()

#         # Backpropagation to calculate the gradients
#         loss.backward()
#         # Update the weights
#         optimizer.step()

#         # Get the count of correct predictions
#         correct += get_correct_prediction_count(pred, target)
#         processed += len(data)

#         # Update the progress bar
#         # msg = f"Progress:\tBatch = {batch_idx} "
#         msg = f"Train: Loss={loss.item():0.4f}, Batch={batch_idx}, Accuracy={100*correct/processed:0.2f}"
#         pbar.set_description(desc=msg)

#     # Close the progress bar
#     pbar.close()

#     # Return the final loss and accuracy for the epoch
#     current_train_accuracy = 100 * correct / processed
#     current_train_loss = train_loss / len(train_loader)

#     # print(f"Training:\tAverage Loss: {current_train_loss:.5f}\tAccuracy: {current_train_accuracy:.2f}%")

#     return current_train_accuracy, current_train_loss


# def test_model(
#     model,
#     device,
#     test_loader,
#     criterion,
#     misclassified_image_data,
#     save_incorrect_predictions=False,
# ):
#     """
#     Function to test the model on the test dataset.
#     """

#     # Initialize the model to evaluation mode
#     model.eval()

#     # Reset the loss and correct predictions for the epoch
#     test_loss = 0
#     correct = 0

#     # Disable gradient calculation while testing
#     with torch.no_grad():
#         for data, target in test_loader:
#             # Move data and labels to device
#             data, target = data.to(device), target.to(device)

#             # Predict using model
#             output = model(data)
#             # Calculate loss for the batch
#             test_loss += criterion(output, target).item()

#             # Get the index of the max log-probability
#             pred = output.argmax(dim=1)
#             # Check if the prediction is correct
#             correct_mask = pred.eq(target)
#             # Save the incorrect predictions
#             incorrect_indices = ~correct_mask

#             # Do this only for last epoch, if not you will run out of memory
#             if save_incorrect_predictions:
#                 # Store images incorrectly predicted, generated predictions and the actual value
#                 misclassified_image_data["images"].extend(data[incorrect_indices])
#                 misclassified_image_data["ground_truths"].extend(target[incorrect_indices])
#                 misclassified_image_data["predicted_vals"].extend(pred[incorrect_indices])

#             # Get the count of correct predictions
#             correct += get_correct_prediction_count(output, target)

#     # Calculate the final loss
#     test_loss /= len(test_loader.dataset)

#     # Return the final loss and accuracy for the epoch
#     current_test_accuracy = 100.0 * correct / len(test_loader.dataset)
#     current_test_loss = test_loss

#     # Print the final test loss and accuracy
#     # print(
#     #     f"Testing:\tAverage Loss: {current_test_loss:.5f}\tAccuracy: {current_test_accuracy:.2f}%",
#     # )
#     print(
#         f"Test set: Average loss: {current_test_loss:.4f}, ",
#         f"Accuracy: {current_test_accuracy:.2f}%",
#     )

#     # Return the final loss and accuracy for the epoch
#     return current_test_accuracy, current_test_loss


# def old_train_and_test_model(
#     batch_size,
#     num_epochs,
#     model,
#     device,
#     train_loader,
#     test_loader,
#     optimizer,
#     criterion,
#     scheduler,
#     misclassified_image_data,
# ):
#     """Trains and tests the model by iterating through epochs"""

#     print(f"\n\nBatch size: {batch_size}, Total epochs: {num_epochs}\n\n")

#     # Hold the results for every epoch
#     results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

#     # Run the model for NUM_EPOCHS
#     for epoch in range(1, num_epochs + 1):
#         # Print the current epoch
#         print(f"Epoch {epoch}")

#         # Train the model
#         epoch_train_accuracy, epoch_train_loss = train_model(model, device, train_loader, optimizer, criterion)

#         # Should we save the incorrect predictions for this epoch?
#         # Do this only for the last epoch, if not you will run out of memory
#         if epoch == num_epochs:
#             save_incorrect_predictions = True
#         else:
#             save_incorrect_predictions = False

#         # Test the model
#         epoch_test_accuracy, epoch_test_loss = test_model(
#             model,
#             device,
#             test_loader,
#             criterion,
#             misclassified_image_data,
#             save_incorrect_predictions,
#         )

#         # Append the train and test accuracies and losses
#         results["train_loss"].append(epoch_train_loss)
#         results["train_acc"].append(epoch_train_accuracy)
#         results["test_loss"].append(epoch_test_loss)
#         results["test_acc"].append(epoch_test_accuracy)

#         # Check if the accuracy is the best accuracy till now
#         # Save the model if you get the best test accuracy
#         if max(results["test_acc"]) == epoch_test_accuracy:
#             # print("Saving the model as best test accuracy till now is achieved!")
#             save_model(
#                 epoch,
#                 model,
#                 optimizer,
#                 scheduler,
#                 batch_size,
#                 criterion,
#                 file_name="model_best_epoch.pth",
#             )

#         # # Passing the latest test loss in list to scheduler to adjust learning rate
#         # scheduler.step(test_losses[-1])
#         scheduler.step()
#         # # # Line break before next epoch
#         print("\n")

#     return results


def train_and_test_model(
    batch_size,
    num_epochs,
    model,
    datamodule,
    logger,
    debug=False,
):
    """Trains and tests the model by iterating through epochs"""

    print(f"\n\nBatch size: {batch_size}, Total epochs: {num_epochs}\n\n")

    # # Hold the results for every epoch
    # results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    checkpoint = ModelCheckpoint(
        dirpath=config.CHECKPOINT_PATH, monitor="val_acc", mode="max", filename="model_best_epoch", save_last=True
    )
    # # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#learningratemonitor
    # lr_rate_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    # Change trainer settings for debugging
    if debug:
        num_epochs = 1
        fast_dev_run = True
        overfit_batches = 0.1
        profiler = "advanced"
    else:
        fast_dev_run = False
        overfit_batches = 0.0
        profiler = None

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#methods
    trainer = pl.Trainer(
        precision=16,
        fast_dev_run=fast_dev_run,
        # deterministic=True,
        # devices="auto",
        # accelerator="auto",
        max_epochs=num_epochs,
        logger=logger,
        # enable_model_summary=False,
        overfit_batches=overfit_batches,
        log_every_n_steps=10,
        # num_sanity_val_steps=5,
        profiler=profiler,
        # check_val_every_n_epoch=1,
        # callbacks=[checkpoint, lr_rate_monitor],
        callbacks=[checkpoint],
    )

    model.lr_finder = model.find_optimal_lr(train_loader=datamodule.train_dataloader())

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    # # Obtain the results dictionary from model
    results = model.results
    # print(f"Results Length: {len(results)}")

    # Get the list of misclassified images
    misclassified_image_data = model.misclassified_image_data
    # print(f"Misclassified Images Length: {len(misclassified_image_data)}")

    return trainer, results, misclassified_image_data
    # return trainer
