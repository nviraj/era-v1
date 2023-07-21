# [Assignment 10](https://canvas.instructure.com/courses/6743641/modules/items/88569751)

## Table of Contents

- [Assignment 10](#assignment-10)
  - [Table of Contents](#table-of-contents)
  - [Assignment Objectives](#assignment-objectives)
  - [Code Overview](#code-overview)
  - [Dataset Details](#dataset-details)
  - [Transformations](#transformations)
  - [Code Overview](#code-overview-1)
  - [Model Parameters](#model-parameters)
  - [Training logs](#training-logs)
  - [Test and Train Metrics](#test-and-train-metrics)
  - [Misclassified Images](#misclassified-images)

<br>

## Assignment Objectives

Write a new network that has

- [x] Target Accuracy: 90%
- [x] Total Epochs = 24
- [x] Uses One Cycle Policy with no annihilation, Max at Epoch 5, Uses LR Finder for Max
- [x] Batch size = 512
- [x] Use ADAM, and CrossEntropyLoss

<br>

## Code Overview

Has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200 pts extra!)

- [x] Total RF must be more than 44
- [x] A Layer must use Depthwise Separable Convolution
- [x] A Layers must use Dilated Convolution
- [x] Use GAP (compulsory)
- [x] Add FC after GAP to target #of classes (optional)
- [x] Use Albumentation library and apply:
  - [x] horizontal flip
  - [x] shiftScaleRotate
  - [x] coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- [x] Achieve 85% accuracy, as many epochs as you want
- [x] Total Params to be less than 200k.
- [x] Follows code-modularity (else 0 for full assignment)

<br>

## Dataset Details

The CIFAR10 dataset is a collection of 60,000 32x32 color images, divided into 50,000 training images and 10,000 test images. The dataset contains 10 classes, each with 6,000 images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The CIFAR10 dataset is available for download from the[website](https://www.cs.toronto.edu/~kriz/cifar.html.) of the Canadian Institute for Advanced Research (CIFAR)

<br>

## Transformations

The following Transformations were applied using the [Albumentations](https://albumentations.ai/) library:

- [HorizontalFlip](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.HorizontalFlip)
- [ShiftScaleRotate](https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/#albumentations.augmentations.geometric.transforms.ShiftScaleRotate)
- [CoarseDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#coarsedropout-augmentation-augmentationsdropoutcoarse_dropout)
- [Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)

Here are some sample images from training data post transformation:
![](<../Files/Transformed Images - Train.png>)

<br>

## Code Overview

We explore various Normalization techniques using Convolution neural networks on CIFAR10 data. To run the code, download the Notebook and modules. Then just run the Notebook and other modules will be automatically imported. The code is structured in a modular way as below:

- **Modules**
  - [dataset.py](modules/dataset.py)
    - Function to download and split CIFAR10 data to test and train - `split_cifar_data()`
    - Function that creates the required test and train transforms compatible with Albumentations - `apply_cifar_image_transformations()`
    - Class that applies the required transforms to dataset - CIFAR10Transforms()
    - Function to calculate mean and standard deviation of the data to normalize tensors - `calculate_mean_std()`
  - [custom_resnet.py](modules/custom_resnet.py)
    - Train and test the model given the optimizer and criterion - `train_model()`, `test_model()`
    - A class called Assignment9 which implements above specified neural network
  - [utils.py](modules/utils.py)
    - Function that detects and returns correct device including GPU and CPU - `get_device()`
    - Given a set of predictions and labels, return the cumulative correct count - `get_correct_prediction_count()`
    - Function to save model, epoch, optimizer, scheduler, loss and batch size - `save_model()`
  - [visualize.py](modules/visualize.py)
    - Given a normalize image along with mean and standard deviation for each channels, convert it back - `convert_back_image()`
    - Plot sample training images along with the labels - `plot_sample_training_images()`
    - Plot train and test metrics - `plot_train_test_metrics()`
    - Plot incorrectly classified images along with ground truth and predicted classes - `plot_misclassified_images()`
- **[Notebook](<ERA V1 - Viraj - Assignment 10.ipynb>)**
  - **Flow**
    - Install and import required libraries
    - Mount Google drive which contains our modules and import them
    - Get device and dataset statistics
    - Apply test and train transformations
    - Split the data to test and train after downloading and applying Transformations
    - Specify the data loader depending on architecture and batch size
    - Define the class labels in a human readable format
    - Display sample images from the training data post transformations
    - Load model to device
    - Show model summary along with tensor size after each block
    - Use LR finder and Once cycle policy
    - Start training and compute various train and test metrics, save best model after each epoch
    - Plot accuracy and loss metrics, also print them in a human readable format
    - Save model after final epoch
    - Show incorrectly predicted images along with actual and predicted labels

<br>

## Model Parameters

**Layer Structure**

```
PrepLayer
	 torch.Size([1, 64, 32, 32])

Layer 1, X
	 torch.Size([1, 128, 16, 16])

Layer 1, R1
	 torch.Size([1, 128, 16, 16])

Layer 1, X + R1
	 torch.Size([1, 128, 16, 16])

Layer 2
	 torch.Size([1, 256, 8, 8])

Layer 3, X
	 torch.Size([1, 512, 4, 4])

Layer 3, R2
	 torch.Size([1, 512, 4, 4])

Layer 3, X + R2
	 torch.Size([1, 512, 4, 4])

Max Pooling
	 torch.Size([1, 512, 1, 1])

Reshape before FC
	 torch.Size([1, 512])

After FC
	 torch.Size([1, 10])

```

**Parameters**

```
========================================================================================================================
Layer (type:depth-idx)                   Input Shape      Kernel Shape     Output Shape     Param #          Trainable
========================================================================================================================
CustomResNet                             [1, 3, 32, 32]   --               [1, 10]          --               True
├─Sequential: 1-1                        [1, 3, 32, 32]   --               [1, 64, 32, 32]  --               True
│    └─Conv2d: 2-1                       [1, 3, 32, 32]   [3, 3]           [1, 64, 32, 32]  1,728            True
│    └─BatchNorm2d: 2-2                  [1, 64, 32, 32]  --               [1, 64, 32, 32]  128              True
│    └─ReLU: 2-3                         [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               --
│    └─Dropout: 2-4                      [1, 64, 32, 32]  --               [1, 64, 32, 32]  --               --
├─Sequential: 1-2                        [1, 64, 32, 32]  --               [1, 128, 16, 16] --               True
│    └─Conv2d: 2-5                       [1, 64, 32, 32]  [3, 3]           [1, 128, 32, 32] 73,728           True
│    └─MaxPool2d: 2-6                    [1, 128, 32, 32] 2                [1, 128, 16, 16] --               --
│    └─BatchNorm2d: 2-7                  [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    └─ReLU: 2-8                         [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Dropout: 2-9                      [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
├─Sequential: 1-3                        [1, 128, 16, 16] --               [1, 128, 16, 16] --               True
│    └─Conv2d: 2-10                      [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    └─BatchNorm2d: 2-11                 [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    └─ReLU: 2-12                        [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Dropout: 2-13                     [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Conv2d: 2-14                      [1, 128, 16, 16] [3, 3]           [1, 128, 16, 16] 147,456          True
│    └─BatchNorm2d: 2-15                 [1, 128, 16, 16] --               [1, 128, 16, 16] 256              True
│    └─ReLU: 2-16                        [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
│    └─Dropout: 2-17                     [1, 128, 16, 16] --               [1, 128, 16, 16] --               --
├─Sequential: 1-4                        [1, 128, 16, 16] --               [1, 256, 8, 8]   --               True
│    └─Conv2d: 2-18                      [1, 128, 16, 16] [3, 3]           [1, 256, 16, 16] 294,912          True
│    └─MaxPool2d: 2-19                   [1, 256, 16, 16] 2                [1, 256, 8, 8]   --               --
│    └─BatchNorm2d: 2-20                 [1, 256, 8, 8]   --               [1, 256, 8, 8]   512              True
│    └─ReLU: 2-21                        [1, 256, 8, 8]   --               [1, 256, 8, 8]   --               --
│    └─Dropout: 2-22                     [1, 256, 8, 8]   --               [1, 256, 8, 8]   --               --
├─Sequential: 1-5                        [1, 256, 8, 8]   --               [1, 512, 4, 4]   --               True
│    └─Conv2d: 2-23                      [1, 256, 8, 8]   [3, 3]           [1, 512, 8, 8]   1,179,648        True
│    └─MaxPool2d: 2-24                   [1, 512, 8, 8]   2                [1, 512, 4, 4]   --               --
│    └─BatchNorm2d: 2-25                 [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    └─ReLU: 2-26                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Dropout: 2-27                     [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
├─Sequential: 1-6                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               True
│    └─Conv2d: 2-28                      [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    └─BatchNorm2d: 2-29                 [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    └─ReLU: 2-30                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Dropout: 2-31                     [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Conv2d: 2-32                      [1, 512, 4, 4]   [3, 3]           [1, 512, 4, 4]   2,359,296        True
│    └─BatchNorm2d: 2-33                 [1, 512, 4, 4]   --               [1, 512, 4, 4]   1,024            True
│    └─ReLU: 2-34                        [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
│    └─Dropout: 2-35                     [1, 512, 4, 4]   --               [1, 512, 4, 4]   --               --
├─MaxPool2d: 1-7                         [1, 512, 4, 4]   4                [1, 512, 1, 1]   --               --
├─Linear: 1-8                            [1, 512]         --               [1, 10]          5,130            True
========================================================================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
Total mult-adds (M): 379.27
========================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 4.65
Params size (MB): 26.29
Estimated Total Size (MB): 30.96
========================================================================================================================

```

<br>

## Training logs

<br>

## Test and Train Metrics

<br>

## Misclassified Images

<br>
