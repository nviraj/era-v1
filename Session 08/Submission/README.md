# [Assignment 8](https://canvas.instructure.com/courses/6743641/quizzes/14668328?module_item_id=87770489)

## Table of Contents

- [Assignment 8](#assignment-8)
  - [Table of Contents](#table-of-contents)
  - [Objectives](#objectives)
  - [Summary of test and train accuracies](#summary-of-test-and-train-accuracies)
  - [Dataset Details](#dataset-details)
  - [Code Overview](#code-overview)
  - [Model](#model)
    - [Architecture/ Code](#architecture-code)
    - [Parameters](#parameters)
    - [Receptive Field](#receptive-field)
  - [Training logs](#training-logs)
    - [Batch Normalization](#batch-normalization)
    - [Group Normalization](#group-normalization)
    - [Layer Normalization](#layer-normalization)
  - [Test and Train Metrics](#test-and-train-metrics)
    - [Batch Normalization](#batch-normalization-1)
    - [Group Normalization](#group-normalization-1)
    - [Layer Normalization](#layer-normalization-1)
  - [Misclassified Images](#misclassified-images)
    - [Batch Normalization](#batch-normalization-2)
    - [Group Normalization](#group-normalization-2)
    - [Layer Normalization](#layer-normalization-2)
  - [Findings](#findings)

<br>

## Objectives

Build a CIFAR10 image classification networks which adheres to the following guidelines:

- C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10
  - Legend: C - Convolution with 3x3 kernels, c - Convolution with 1x1 kernels, P - Padding, GAP - Global Average Pooling
- Max Epochs 20, Max Parameters 50k
- Minimum Accuracy of 70% in below scenarios
  - Network with Group Normalization
  - Network with Layer Normalization
  - Network with Batch Normalization

<br>

## Summary of test and train accuracies

| Normalization Technique | Test Accuracy (Max) | Train Accuracy (Max) |
| ----------------------- | ------------------- | -------------------- |
| Batch Normalization     | 78.95%              | 73.69%               |
| Group Normalization     | 78.75%              | 76.47%               |
| Layer Normalization     | 78.29%              | 75.87%               |

<br>

## Dataset Details

The CIFAR10 dataset is a collection of 60,000 32x32 color images, divided into 50,000 training images and 10,000 test images. The dataset contains 10 classes, each with 6,000 images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The CIFAR10 dataset is available for download from the [website](https://www.cs.toronto.edu/~kriz/cifar.html.) of the Canadian Institute for Advanced Research (CIFAR):

<br>

## Code Overview

We explore various Normalization techniques using Convolution neural networks on CIFAR10 data. The code is structured in a modular way as below:

- **Modules**
  - [dataset.py](dataset.py)
    - Function to download and split CIFAR10 data to test and train
    - Function to calculate mean and standard deviation of the data to normalize tensors
  - [model.py](model.py)
    - Train and test the model given the optimizer and criterion
    - A class called NormalizationModel which implements above specified neural network
      - This accepts the Normalization method and number of groups and parameters and for each block applies the appropriate Normalization technique
  - [utils.py](utils.py)
    - Function that detects and returns correct device including GPU and CPU
    - Given a set of predictions and labels, return the cumulative correct count
  - [visualize.py](visualize.py)
    - Given a normalize image along with mean and standard deviation for each channels, convert it back
    - Plot sample training images along with the labels
    - Plot train and test metrics
    - Plot incorrectly classified images along with ground truth and predicted classes
- **Notebooks**
  - **Flow**
    - Install and import required libraries
    - Mount Google drive which contains our modules and import them
    - Get device and dataset statistics
    - Apply test and train transformations
    - Split the data to test and train after downloading and applying Transformations
    - Specify the data loader depending on architecture and batch size
    - Define the class labels in a human readable format
    - Display sample images from the training data
    - Load model to device
    - Show model summary along with tensor size after each block
    - Start training and Compute various train and test metrics
    - Plot accuracy and loss metrics
    - Save model
    - Show incorrectly predicted images along with actual and predicted labels
  - **Files**
    - [ERA V1 - Viraj - Assignment 08 - Batch Normalization.ipynb](<ERA V1 - Viraj - Assignment 08 - Batch Normalization.ipynb>)
      - Batch Normalization has been applied using nn.BatchNorm2d()
    - [ERA V1 - Viraj - Assignment 08 - Group Normalization.ipynb](<ERA V1 - Viraj - Assignment 08 - Group Normalization.ipynb>)
      - Group Normalization has been applied using nn.GroupNorm(num_group)
    - [ERA V1 - Viraj - Assignment 08 - Layer Normalization.ipynb](<ERA V1 - Viraj - Assignment 08 - Layer Normalization.ipynb>)
      - Layer Normalization has been applied using nn.GroupNorm(1)

<br>

## Model

### Architecture/ Code

```
# This is for Assignment 8
class NormalizationModel(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.05

    def __init__(self, normalization_method="batch", num_groups=2):
        super().__init__()

        self.norm = normalization_method
        self.num_group = num_groups

        # Throw an error if normalization method does not match list of acceptable values
        allowed_norm = ["batch", "layer", "group"]
        if self.norm not in allowed_norm:
            raise ValueError(f"Normalization method must be one of {allowed_norm}")

        # if layer normalisation is chosen change num_groups to 1
        if self.norm == "layer":
            self.num_group = 1

        # if group normalisation is chosen throw an error if num_groups is less than 2
        if self.norm == "group":
            if self.num_group < 2:
                raise ValueError(
                    "Number of groups must be greater than 1 for group normalization"
                )

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 12),
        )

        self.C2 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 16),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(8)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 8),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.C3 = nn.Sequential(
            nn.Conv2d(8, 24, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 24),
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 24),
        )

        self.C5 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 24),
        )

        self.c6 = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=(1, 1), stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 12),
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.C7 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 32),
        )

        self.C8 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32)
            if self.norm == "batch"
            else nn.GroupNorm(self.num_group, 32),
        )

        self.C9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.c10 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=(1, 1), stride=1, bias=False)
        )

    def print_view(self, x):
        """Print shape of the model"""
        if self.print_shape:
            print(x.shape)

    def forward(self, x):
        """Forward pass"""

        x = self.C1(x)
        self.print_view(x)
        x = self.C2(x)
        self.print_view(x)
        x = self.c3(x)
        self.print_view(x)
        x = self.pool1(x)
        x = self.C3(x)
        self.print_view(x)
        x = self.C4(x)
        self.print_view(x)
        x = self.C4(x)
        self.print_view(x)
        x = self.C5(x)
        self.print_view(x)
        x = self.c6(x)
        self.print_view(x)
        x = self.pool2(x)
        self.print_view(x)
        x = self.C7(x)
        self.print_view(x)
        x = self.C8(x)
        self.print_view(x)
        x = self.C9(x)
        self.print_view(x)
        x = self.gap(x)
        self.print_view(x)
        x = self.c10(x)
        self.print_view(x)
        x = x.view((x.shape[0], -1))
        self.print_view(x)
        x = F.log_softmax(x, dim=1)

        return x
```

### Parameters

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 32, 32]             324
              ReLU-2           [-1, 12, 32, 32]               0
           Dropout-3           [-1, 12, 32, 32]               0
       BatchNorm2d-4           [-1, 12, 32, 32]              24
            Conv2d-5           [-1, 16, 32, 32]           1,728
              ReLU-6           [-1, 16, 32, 32]               0
           Dropout-7           [-1, 16, 32, 32]               0
       BatchNorm2d-8           [-1, 16, 32, 32]              32
            Conv2d-9            [-1, 8, 32, 32]             128
             ReLU-10            [-1, 8, 32, 32]               0
          Dropout-11            [-1, 8, 32, 32]               0
      BatchNorm2d-12            [-1, 8, 32, 32]              16
        MaxPool2d-13            [-1, 8, 16, 16]               0
           Conv2d-14           [-1, 24, 16, 16]           1,728
             ReLU-15           [-1, 24, 16, 16]               0
          Dropout-16           [-1, 24, 16, 16]               0
      BatchNorm2d-17           [-1, 24, 16, 16]              48
           Conv2d-18           [-1, 24, 16, 16]           5,184
             ReLU-19           [-1, 24, 16, 16]               0
          Dropout-20           [-1, 24, 16, 16]               0
      BatchNorm2d-21           [-1, 24, 16, 16]              48
           Conv2d-22           [-1, 24, 16, 16]           5,184
             ReLU-23           [-1, 24, 16, 16]               0
          Dropout-24           [-1, 24, 16, 16]               0
      BatchNorm2d-25           [-1, 24, 16, 16]              48
           Conv2d-26           [-1, 24, 16, 16]           5,184
             ReLU-27           [-1, 24, 16, 16]               0
          Dropout-28           [-1, 24, 16, 16]               0
      BatchNorm2d-29           [-1, 24, 16, 16]              48
           Conv2d-30           [-1, 12, 16, 16]             288
             ReLU-31           [-1, 12, 16, 16]               0
          Dropout-32           [-1, 12, 16, 16]               0
      BatchNorm2d-33           [-1, 12, 16, 16]              24
        MaxPool2d-34             [-1, 12, 8, 8]               0
           Conv2d-35             [-1, 32, 8, 8]           3,456
             ReLU-36             [-1, 32, 8, 8]               0
          Dropout-37             [-1, 32, 8, 8]               0
      BatchNorm2d-38             [-1, 32, 8, 8]              64
           Conv2d-39             [-1, 32, 8, 8]           9,216
             ReLU-40             [-1, 32, 8, 8]               0
          Dropout-41             [-1, 32, 8, 8]               0
      BatchNorm2d-42             [-1, 32, 8, 8]              64
           Conv2d-43             [-1, 32, 8, 8]           9,216
AdaptiveAvgPool2d-44             [-1, 32, 1, 1]               0
           Conv2d-45             [-1, 10, 1, 1]             320
================================================================
Total params: 42,372
Trainable params: 42,372
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.13
Params size (MB): 0.16
Estimated Total Size (MB): 2.30
----------------------------------------------------------------

```

### Receptive Field

<br>

## Training logs

### Batch Normalization

```
Batch size: 64, Total epochs: 20


Epoch 1
Train: Loss=1.3748, Batch_id=781, Accuracy=44.06: 100%|██████████| 782/782 [00:22<00:00, 34.45it/s]
Test set: Average loss: 1.2349, Accuracy: 5487/10000 (54.87%)


Epoch 2
Train: Loss=0.8731, Batch_id=781, Accuracy=58.10: 100%|██████████| 782/782 [00:21<00:00, 36.32it/s]
Test set: Average loss: 1.0627, Accuracy: 6173/10000 (61.73%)


Epoch 3
Train: Loss=1.0034, Batch_id=781, Accuracy=62.46: 100%|██████████| 782/782 [00:20<00:00, 37.41it/s]
Test set: Average loss: 1.0191, Accuracy: 6324/10000 (63.24%)


Epoch 4
Train: Loss=1.0314, Batch_id=781, Accuracy=65.54: 100%|██████████| 782/782 [00:21<00:00, 35.74it/s]
Test set: Average loss: 0.9389, Accuracy: 6654/10000 (66.54%)


Epoch 5
Train: Loss=1.2885, Batch_id=781, Accuracy=67.12: 100%|██████████| 782/782 [00:22<00:00, 35.33it/s]
Test set: Average loss: 0.9120, Accuracy: 6796/10000 (67.96%)


Epoch 6
Train: Loss=0.8926, Batch_id=781, Accuracy=68.93: 100%|██████████| 782/782 [00:20<00:00, 37.89it/s]
Test set: Average loss: 0.8524, Accuracy: 7040/10000 (70.40%)


Epoch 7
Train: Loss=0.9119, Batch_id=781, Accuracy=70.49: 100%|██████████| 782/782 [00:20<00:00, 37.89it/s]
Test set: Average loss: 0.8173, Accuracy: 7161/10000 (71.61%)


Epoch 8
Train: Loss=0.4851, Batch_id=781, Accuracy=71.70: 100%|██████████| 782/782 [00:24<00:00, 32.24it/s]
Test set: Average loss: 0.8545, Accuracy: 6992/10000 (69.92%)


Epoch 9
Train: Loss=0.7880, Batch_id=781, Accuracy=72.70: 100%|██████████| 782/782 [00:22<00:00, 35.31it/s]
Test set: Average loss: 0.8380, Accuracy: 7064/10000 (70.64%)


Epoch 10
Train: Loss=1.3480, Batch_id=781, Accuracy=74.91: 100%|██████████| 782/782 [00:20<00:00, 38.25it/s]
Test set: Average loss: 0.7597, Accuracy: 7393/10000 (73.93%)


Epoch 11
Train: Loss=0.5559, Batch_id=781, Accuracy=75.48: 100%|██████████| 782/782 [00:21<00:00, 37.15it/s]
Test set: Average loss: 0.7946, Accuracy: 7226/10000 (72.26%)


Epoch 12
Train: Loss=0.7195, Batch_id=781, Accuracy=75.60: 100%|██████████| 782/782 [00:21<00:00, 35.55it/s]
Test set: Average loss: 0.8663, Accuracy: 7046/10000 (70.46%)


Epoch 13
Train: Loss=0.3051, Batch_id=781, Accuracy=77.18: 100%|██████████| 782/782 [00:21<00:00, 37.19it/s]
Test set: Average loss: 0.8283, Accuracy: 7207/10000 (72.07%)


Epoch 14
Train: Loss=1.0073, Batch_id=781, Accuracy=77.52: 100%|██████████| 782/782 [00:21<00:00, 37.20it/s]
Test set: Average loss: 0.8242, Accuracy: 7207/10000 (72.07%)


Epoch 15
Train: Loss=0.9650, Batch_id=781, Accuracy=78.10: 100%|██████████| 782/782 [00:22<00:00, 34.78it/s]
Test set: Average loss: 0.7689, Accuracy: 7369/10000 (73.69%)


Epoch 16
Train: Loss=0.9035, Batch_id=781, Accuracy=78.10: 100%|██████████| 782/782 [00:24<00:00, 32.16it/s]
Test set: Average loss: 0.7924, Accuracy: 7268/10000 (72.68%)


Epoch 17
Train: Loss=0.9131, Batch_id=781, Accuracy=78.33: 100%|██████████| 782/782 [00:21<00:00, 35.70it/s]
Test set: Average loss: 0.7821, Accuracy: 7330/10000 (73.30%)


Epoch 18
Train: Loss=0.6066, Batch_id=781, Accuracy=78.42: 100%|██████████| 782/782 [00:21<00:00, 37.21it/s]
Test set: Average loss: 0.7874, Accuracy: 7318/10000 (73.18%)


Epoch 19
Train: Loss=0.6265, Batch_id=781, Accuracy=78.52: 100%|██████████| 782/782 [00:21<00:00, 35.75it/s]
Test set: Average loss: 0.8052, Accuracy: 7267/10000 (72.67%)


Epoch 20
Train: Loss=1.0927, Batch_id=781, Accuracy=78.95: 100%|██████████| 782/782 [00:23<00:00, 33.77it/s]
Test set: Average loss: 0.7738, Accuracy: 7351/10000 (73.51%)
```

### Group Normalization

```
Batch size: 64, Total epochs: 20


Epoch 1
Train: Loss=1.3367, Batch_id=781, Accuracy=36.29: 100%|██████████| 782/782 [00:25<00:00, 30.11it/s]
Test set: Average loss: 1.4137, Accuracy: 4754/10000 (47.54%)


Epoch 2
Train: Loss=0.8531, Batch_id=781, Accuracy=52.66: 100%|██████████| 782/782 [00:24<00:00, 32.06it/s]
Test set: Average loss: 1.2109, Accuracy: 5734/10000 (57.34%)


Epoch 3
Train: Loss=1.0642, Batch_id=781, Accuracy=59.28: 100%|██████████| 782/782 [00:22<00:00, 34.22it/s]
Test set: Average loss: 1.0541, Accuracy: 6217/10000 (62.17%)


Epoch 4
Train: Loss=1.0248, Batch_id=781, Accuracy=63.14: 100%|██████████| 782/782 [00:22<00:00, 35.05it/s]
Test set: Average loss: 1.0726, Accuracy: 6268/10000 (62.68%)


Epoch 5
Train: Loss=1.0772, Batch_id=781, Accuracy=66.29: 100%|██████████| 782/782 [00:22<00:00, 34.27it/s]
Test set: Average loss: 0.9191, Accuracy: 6782/10000 (67.82%)


Epoch 6
Train: Loss=0.7634, Batch_id=781, Accuracy=68.34: 100%|██████████| 782/782 [00:24<00:00, 31.99it/s]
Test set: Average loss: 0.8742, Accuracy: 6868/10000 (68.68%)


Epoch 7
Train: Loss=0.8296, Batch_id=781, Accuracy=70.02: 100%|██████████| 782/782 [00:25<00:00, 30.78it/s]
Test set: Average loss: 0.8544, Accuracy: 6993/10000 (69.93%)


Epoch 8
Train: Loss=0.6838, Batch_id=781, Accuracy=71.23: 100%|██████████| 782/782 [00:25<00:00, 30.12it/s]
Test set: Average loss: 0.8582, Accuracy: 7028/10000 (70.28%)


Epoch 9
Train: Loss=0.7436, Batch_id=781, Accuracy=73.89: 100%|██████████| 782/782 [00:25<00:00, 30.73it/s]
Test set: Average loss: 0.7615, Accuracy: 7369/10000 (73.69%)


Epoch 10
Train: Loss=1.3788, Batch_id=781, Accuracy=74.67: 100%|██████████| 782/782 [00:25<00:00, 30.69it/s]
Test set: Average loss: 0.7436, Accuracy: 7358/10000 (73.58%)


Epoch 11
Train: Loss=0.6470, Batch_id=781, Accuracy=75.01: 100%|██████████| 782/782 [00:23<00:00, 33.56it/s]
Test set: Average loss: 0.7415, Accuracy: 7379/10000 (73.79%)


Epoch 12
Train: Loss=0.6102, Batch_id=781, Accuracy=76.73: 100%|██████████| 782/782 [00:22<00:00, 34.96it/s]
Test set: Average loss: 0.7100, Accuracy: 7529/10000 (75.29%)


Epoch 13
Train: Loss=0.2736, Batch_id=781, Accuracy=76.66: 100%|██████████| 782/782 [00:22<00:00, 34.51it/s]
Test set: Average loss: 0.7109, Accuracy: 7511/10000 (75.11%)


Epoch 14
Train: Loss=0.8833, Batch_id=781, Accuracy=77.04: 100%|██████████| 782/782 [00:24<00:00, 31.38it/s]
Test set: Average loss: 0.7017, Accuracy: 7515/10000 (75.15%)


Epoch 15
Train: Loss=0.8418, Batch_id=781, Accuracy=77.65: 100%|██████████| 782/782 [00:23<00:00, 32.84it/s]
Test set: Average loss: 0.6861, Accuracy: 7604/10000 (76.04%)


Epoch 16
Train: Loss=0.4639, Batch_id=781, Accuracy=77.93: 100%|██████████| 782/782 [00:23<00:00, 33.08it/s]
Test set: Average loss: 0.6857, Accuracy: 7605/10000 (76.05%)


Epoch 17
Train: Loss=0.7710, Batch_id=781, Accuracy=78.02: 100%|██████████| 782/782 [00:22<00:00, 34.06it/s]
Test set: Average loss: 0.6852, Accuracy: 7611/10000 (76.11%)


Epoch 18
Train: Loss=0.6340, Batch_id=781, Accuracy=78.63: 100%|██████████| 782/782 [00:22<00:00, 35.17it/s]
Test set: Average loss: 0.6866, Accuracy: 7592/10000 (75.92%)


Epoch 19
Train: Loss=0.8444, Batch_id=781, Accuracy=78.58: 100%|██████████| 782/782 [00:21<00:00, 35.68it/s]
Test set: Average loss: 0.6752, Accuracy: 7639/10000 (76.39%)


Epoch 20
Train: Loss=0.7628, Batch_id=781, Accuracy=78.75: 100%|██████████| 782/782 [00:24<00:00, 32.32it/s]
Test set: Average loss: 0.6782, Accuracy: 7647/10000 (76.47%)
```

### Layer Normalization

```
Batch size: 64, Total epochs: 20


Epoch 1
Train: Loss=1.2331, Batch_id=781, Accuracy=36.81: 100%|██████████| 782/782 [00:22<00:00, 35.20it/s]
Test set: Average loss: 1.6171, Accuracy: 4117/10000 (41.17%)


Epoch 2
Train: Loss=0.8703, Batch_id=781, Accuracy=51.53: 100%|██████████| 782/782 [00:21<00:00, 35.75it/s]
Test set: Average loss: 1.2163, Accuracy: 5623/10000 (56.23%)


Epoch 3
Train: Loss=1.0694, Batch_id=781, Accuracy=58.30: 100%|██████████| 782/782 [00:21<00:00, 35.88it/s]
Test set: Average loss: 1.0585, Accuracy: 6182/10000 (61.82%)


Epoch 4
Train: Loss=0.9385, Batch_id=781, Accuracy=62.05: 100%|██████████| 782/782 [00:21<00:00, 37.14it/s]
Test set: Average loss: 1.1184, Accuracy: 6047/10000 (60.47%)


Epoch 5
Train: Loss=0.9736, Batch_id=781, Accuracy=64.44: 100%|██████████| 782/782 [00:22<00:00, 34.89it/s]
Test set: Average loss: 0.9722, Accuracy: 6550/10000 (65.50%)


Epoch 6
Train: Loss=0.4555, Batch_id=781, Accuracy=67.32: 100%|██████████| 782/782 [00:23<00:00, 33.89it/s]
Test set: Average loss: 0.8778, Accuracy: 6954/10000 (69.54%)


Epoch 7
Train: Loss=0.7794, Batch_id=781, Accuracy=68.58: 100%|██████████| 782/782 [00:21<00:00, 37.04it/s]
Test set: Average loss: 0.8735, Accuracy: 6949/10000 (69.49%)


Epoch 8
Train: Loss=0.5288, Batch_id=781, Accuracy=70.22: 100%|██████████| 782/782 [00:22<00:00, 35.27it/s]
Test set: Average loss: 0.9184, Accuracy: 6828/10000 (68.28%)


Epoch 9
Train: Loss=1.0254, Batch_id=781, Accuracy=73.26: 100%|██████████| 782/782 [00:22<00:00, 34.95it/s]
Test set: Average loss: 0.7815, Accuracy: 7252/10000 (72.52%)


Epoch 10
Train: Loss=1.5264, Batch_id=781, Accuracy=73.68: 100%|██████████| 782/782 [00:22<00:00, 34.07it/s]
Test set: Average loss: 0.7593, Accuracy: 7350/10000 (73.50%)


Epoch 11
Train: Loss=0.5385, Batch_id=781, Accuracy=74.19: 100%|██████████| 782/782 [00:22<00:00, 34.70it/s]
Test set: Average loss: 0.7919, Accuracy: 7245/10000 (72.45%)


Epoch 12
Train: Loss=0.5697, Batch_id=781, Accuracy=75.57: 100%|██████████| 782/782 [00:21<00:00, 36.97it/s]
Test set: Average loss: 0.7381, Accuracy: 7440/10000 (74.40%)


Epoch 13
Train: Loss=0.3819, Batch_id=781, Accuracy=76.07: 100%|██████████| 782/782 [00:21<00:00, 36.40it/s]
Test set: Average loss: 0.7198, Accuracy: 7485/10000 (74.85%)


Epoch 14
Train: Loss=0.7962, Batch_id=781, Accuracy=76.24: 100%|██████████| 782/782 [00:22<00:00, 34.20it/s]
Test set: Average loss: 0.7238, Accuracy: 7485/10000 (74.85%)


Epoch 15
Train: Loss=1.1932, Batch_id=781, Accuracy=77.06: 100%|██████████| 782/782 [00:22<00:00, 34.47it/s]
Test set: Average loss: 0.7142, Accuracy: 7501/10000 (75.01%)


Epoch 16
Train: Loss=0.4948, Batch_id=781, Accuracy=77.38: 100%|██████████| 782/782 [00:21<00:00, 36.59it/s]
Test set: Average loss: 0.7035, Accuracy: 7538/10000 (75.38%)


Epoch 17
Train: Loss=0.6874, Batch_id=781, Accuracy=77.48: 100%|██████████| 782/782 [00:21<00:00, 36.37it/s]
Test set: Average loss: 0.7041, Accuracy: 7572/10000 (75.72%)


Epoch 18
Train: Loss=0.6296, Batch_id=781, Accuracy=78.16: 100%|██████████| 782/782 [00:22<00:00, 34.29it/s]
Test set: Average loss: 0.7054, Accuracy: 7558/10000 (75.58%)


Epoch 19
Train: Loss=0.8874, Batch_id=781, Accuracy=78.27: 100%|██████████| 782/782 [00:22<00:00, 34.03it/s]
Test set: Average loss: 0.6943, Accuracy: 7581/10000 (75.81%)


Epoch 20
Train: Loss=1.0406, Batch_id=781, Accuracy=78.29: 100%|██████████| 782/782 [00:21<00:00, 36.72it/s]
Test set: Average loss: 0.6970, Accuracy: 7587/10000 (75.87%)
```

<br>

## Test and Train Metrics

### Batch Normalization

![](<../Files/Normalization - Batch - Metrics.png>)

### Group Normalization

![](<../Files/Normalization - Group - Metrics.png>)

### Layer Normalization

![](<../Files/Normalization - Layer - Metrics.png>)

<br>

## Misclassified Images

### Batch Normalization

![](<../Files/Normalization - Batch - Misclassified.png>)

### Group Normalization

![](<../Files/Normalization - Group - Misclassified.png>)

### Layer Normalization

![](<../Files/Normalization - Layer - Misclassified.png>)

<br>

## Findings
