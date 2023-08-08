"""Module to define the train and test functions."""

# from functools import partial

import modules.config as config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

# Import tuner
from pytorch_lightning.tuner.tuning import Tuner

# What is the start LR and weight decay you'd prefer?
PREFERRED_START_LR = config.PREFERRED_START_LR


def train_and_test_model(
    batch_size,
    num_epochs,
    model,
    datamodule,
    logger,
    debug=False,
):
    """Trains and tests the model by iterating through epochs using Lightning Trainer."""

    print(f"\n\nBatch size: {batch_size}, Total epochs: {num_epochs}\n\n")

    print("Defining Lightning Callbacks")

    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    checkpoint = ModelCheckpoint(
        dirpath=config.CHECKPOINT_PATH, monitor="val_acc", mode="max", filename="model_best_epoch", save_last=True
    )
    # # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#learningratemonitor
    lr_rate_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=False)
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelSummary.html#lightning.pytorch.callbacks.ModelSummary
    model_summary = ModelSummary(max_depth=0)

    print("Defining Lightning Trainer")
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
        callbacks=[checkpoint, lr_rate_monitor, model_summary],
        # callbacks=[checkpoint],
    )

    # # Using the learning rate finder
    # model.learning_rate = model.find_optimal_lr(train_loader=datamodule.train_dataloader())

    # Using the lr_find from Trainer.tune method instead
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner
    # https://www.youtube.com/watch?v=cLZv0eZQSIE
    print("Finding the optimal learning rate using Lightning Tuner.")
    tuner = Tuner(trainer)
    tuner.lr_find(
        model=model,
        datamodule=datamodule,
        min_lr=PREFERRED_START_LR,
        max_lr=5,
        num_training=200,
        mode="linear",
        early_stop_threshold=10,
        attr_name="learning_rate",
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, dataloaders=datamodule.test_dataloader())

    # # Obtain the results dictionary from model
    print("Collecting epoch level model results.")
    results = model.results
    # print(f"Results Length: {len(results)}")

    # Get the list of misclassified images
    print("Collecting misclassified images.")
    misclassified_image_data = model.misclassified_image_data
    # print(f"Misclassified Images Length: {len(misclassified_image_data)}")

    return trainer, results, misclassified_image_data
    # return trainer
