# [Assignment 8](https://canvas.instructure.com/courses/6743641/quizzes/14668328?module_item_id=87770489)

## Table of Contents

## Objectives

Build a CIFAR10 image classification networks which adheres to the following guidelines:

- C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10
  - Legend: C - Convolution with 3x3 kernels, c - Convolution with 1x1 kernels, P - Padding, GAP - Global Average Pooling
- Max Epochs 20, Max Parameters 50k
- Minimum Accuracy of 70% in below scenarios
  - Network with Group Normalization
  - Network with Layer Normalization
  - Network with Batch Normalization

## Summary of test and train accuracies

| Normalization Technique | Test Accuracy (Max) | Train Accuracy (Max) |
| ----------------------- | ------------------- | -------------------- |
| Batch Normalization     | 78.95%              | 73.69%               |
| Group Normalization     | 78.75%              | 76.47%               |
| Layer Normalization     | 78.29%              | 75.87%               |

## Dataset Details

The CIFAR10 dataset is a collection of 60,000 32x32 color images, divided into 50,000 training images and 10,000 test images. The dataset contains 10 classes, each with 6,000 images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The CIFAR10 dataset is available for download from the [website](https://www.cs.toronto.edu/~kriz/cifar.html.) of the Canadian Institute for Advanced Research (CIFAR):

## Code Overview

We explore various Normalization techniques using Convolution neural networks on CIFAR10 data. The code is structured in a modular way as below:

- Modules
  - py
    - Function to download and split CIFAR10 data to test and train
    - Function to calculate mean and standard deviation of the data to normalize tensors
  - py
    - Train and test the model given the optimizer and criterion
    - A class called NormalizationModel which implements above specified neural network
      - This accepts the Normalization method and number of groups and parameters and for each block applies the appropriate Normalization technique
  - py
    - Function that detects and returns correct device including GPU and CPU
    - Given a set of predictions and labels, return the cumulative correct count
  - py
    - Given a normalize image along with mean and standard deviation for each channels, convert it back
    - Plot sample training images along with the labels
    - Plot train and test metrics
    - Plot incorrectly classified images along with ground truth and predicted classes
- Notebooks
  - Flow
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
  - ERA V1 - Viraj - Assignment 08 - Batch Normalization.ipynb
    - Batch Normalization has been applied using nn.BatchNorm2d()
  - ERA V1 - Viraj - Assignment 08 - Group Normalization.ipynb
    - Group Normalization has been applied using nn.GroupNorm(num_group)
  - ERA V1 - Viraj - Assignment 08 - Layer Normalization.ipynb
    - Layer Normalization has been applied using nn.GroupNorm(1)

## Model

### Architecture/ Code

\<Add details of 1 class\>

### Parameters

### Receptive Field

## Training logs

### Batch Normalization

### Group Normalization

### Layer Normalization

## Test and Train Metrics

### Batch Normalization

### Group Normalization

### Layer Normalization

## Misclassified Images

### Batch Normalization

### Group Normalization

### Layer Normalization

## Findings
