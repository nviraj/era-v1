# [Assignment 07](https://canvas.instructure.com/courses/6743641/quizzes/14581494?module_item_id=87206532)

## Table of contents

- [Assignment 07](#assignment-07)
  - [Table of contents](#table-of-contents)
  - [Objectives](#objectives)
  - [Experiments](#experiments)
    - [Model 01](#model-01)
    - [Model 02](#model-02)
    - [Model 03](#model-03)
    - [Model 04](#model-04)
    - [Model 05](#model-05)
    - [Model 06](#model-06)
  - [Model Details (Last Iteration)](#model-details-last-iteration)
  - [Results (Last Iteration)](#results-last-iteration)
  - [Train and Test Metrics (Last Iteration)](#train-and-test-metrics-last-iteration)
  - [Receptive Field Calculation (Last Iteration)](#receptive-field-calculation-last-iteration)

<br>

## Objectives

Build a MNIST image classification CNN which adheres to the following guidelines:

- Test Accuracy of \>99.4% (This must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters

<br>

## Experiments

The objective was achieved using an iterative approach. Models 1 to 6 were developed incrementally in the below manner. The class definition of the CNN can be found in the [model.py](model.py). Commonly used utilites in notebooks can be found in [utils.py](utils.py)

### Model 01

- Target
  - Get a functional skeleton of the neural network working which allows for flexibility and scope for improvement unlike Mercedes's F1 car from the last 2 years
  - Apart from Convolution, don't plan on using Fully connected layers and keep a placeholder for Gap layer
- Results
  - Parameters: 75,024
  - Best train accuracy: 99.25%
  - Best test accuracy: 99.24%
- Analyses
  - The model is large in terms of parameters
  - Each Convolution layer is followed by a Relu function
  - There are 6 Convolution layers followed an adaptive average pooling layer
  - For the size of the model accuracy is not impressive, but we are not trying to optimize in any manner
- [Notebook](<ERA V1 - Viraj - Assignment 07 - Model 01.ipynb>)

### Model 02

- Target
  - Reduce number of parameters by adjusting the number of channels in each layer
  - Introduce max pooling
  - Add a class variable called print_shape which will make debugging the input and output size after each block used with sequential syntax in PyTorch
- Results
  - Parameters: 8,136
  - Best train accuracy: 99.20%
  - Best test accuracy: 99.10%
- Analyses
  - The number of parameters has dropped significantly compared to Model 1
  - For the reduction in parameters, the corresponding reduction in accuracy feels acceptable
  - There is no over fitting
  - Initially a max pooling layer was placed after both block 2 and block 3 but the reduction in image size tended to be high
  - The model has scope for pushing and the skeleton feels alright
- [Notebook](<ERA V1 - Viraj - Assignment 07 - Model 02.ipynb>)

### Model 03

- Target
  - Introduce batch normalization to increase model efficiency
    - Added after every Convolution layer
  - Introduce padding
    - Added only in input layer
  - Reduced channels in block 3 to bring the total number of parameters below 8000
  - Introduce a class method called print_view which prints the shapes of the blocks conditionally based on class variable print_shape introduced in previous model
- Results
  - Parameters: 7,880
  - Best train accuracy: 99.64%
  - Best test accuracy: 99.32%
- Analyses
  - Both test and train accuracy saw an improvement as a result of batch normalization
  - Total number of parameters is now below the required count
  - The difference between test and train accuracy is not showing tendency to overfit
  - Further optimisation of the model is required to push test accuracy higher
- [Notebook](<ERA V1 - Viraj - Assignment 07 - Model 03.ipynb>)

### Model 04

- Target
  - Introduce dropouts of 0.1
    - Skipped first layer to gain as much information as possible in the starting phase of the network
    - Dropout was avoided in the output as well as penultimate layer
    - Additionally dropout was also skipped in the layer with max pooling
- Results
  - Parameters: 7,880
  - Best train accuracy: 99.30%
  - Best test accuracy: 99.42%
- Analyses
  - We have finally breached the 99.4% test accuracy even though it happened in only 2 layers
  - Regularization has also reduced training accuracy compared to previous model but this is to be expected as tendency to overfit will be curbed
  - We also need to explore ways to increase accuracy in subsequent models by either augmenting data quality or tweaking existing parameters such as dropout factor
- [Notebook](<ERA V1 - Viraj - Assignment 07 - Model 04.ipynb>)

### Model 05

- Target
  - Apply data Transformations such as rotation
    - A rotation of 6 degrees was applied either ways and filled with white values
  - Adjust dropout value and see it's impact on accuracy
- Results
  - Parameters: 7,880
  - Best train accuracy: 99.28%
  - Best test accuracy: 99.43%
- Analyses
  - Increasing dropout to 0.15 saw a reduction in accuracy whereas lowering it to 0.05 saw the same level of accuracy as previous model
  - Model is learning well but the test accuracy is still not consistently above 99.4
  - Tweaking the model learning rate might also show good results as we are reaching limits of the model without changing the skeleton and sticking to the parameter limit
- [Notebook](<ERA V1 - Viraj - Assignment 07 - Model 05.ipynb>)

### Model 06

- Target
  - Adjust learning rate of the model when the test loss plateaus or increases
    - We use ReduceLROnPlateau instead of StepLR which has more guesswork involved
  - Reduce dropout even further to 0.01 from 0.05 in previous mode
- Results
  - Parameters: 7,880
  - Best train accuracy: 99.56%
  - Best test accuracy: 99.47%
- Analyses
  - Adjusting both dropout and learning rates in conjunction with all the incremental updates has finally paid dividend
  - From epoch 9 to 15 (With exception of 10th epoch which still has 99.39%), the test accuracy is above 99.4% which is great
  - Building the model step by step helps from losing direction and also saves time
- [Notebook](<ERA V1 - Viraj - Assignment 07 - Model 06.ipynb>)

<br>

## Model Details (Last Iteration)

```
# This is for Model 06
class Model06(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False

    def __init__(self):
        super().__init__()

        # Block 1 - Input Block
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01),
            nn.Conv2d(
                in_channels=10,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.01),
        )

        # Block 4 - Don't use Max Pooling here
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=14,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.01),
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        # Block 5 - Output Block
        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.AdaptiveAvgPool2d(1),
        )

    def print_view(self, x):
        """Print shape of the model"""
        if self.print_shape:
            print(x.shape)

    def forward(self, x):
        """Forward pass"""
        x = self.block1(x)
        self.print_view(x)
        x = self.block2(x)
        self.print_view(x)
        x = self.block3(x)
        self.print_view(x)
        x = self.block4(x)
        self.print_view(x)
        x = self.block5(x)
        self.print_view(x)
        x = x.view((x.shape[0], -1))
        self.print_view(x)
        return F.log_softmax(x, dim=-1)
```

<br>

## Results (Last Iteration)

```
Batch size: 32, Total epochs: 15


Epoch 1
Train: Loss=0.0442, Batch_id=1874, Accuracy=96.16: 100%|██████████| 1875/1875 [00:35<00:00, 52.21it/s]
Test set: Average loss: 0.0451, Accuracy: 9852/10000 (98.52%)


Epoch 2
Train: Loss=0.0271, Batch_id=1874, Accuracy=98.36: 100%|██████████| 1875/1875 [00:34<00:00, 53.88it/s]
Test set: Average loss: 0.0346, Accuracy: 9878/10000 (98.78%)


Epoch 3
Train: Loss=0.0249, Batch_id=1874, Accuracy=98.69: 100%|██████████| 1875/1875 [00:37<00:00, 50.46it/s]
Test set: Average loss: 0.0350, Accuracy: 9886/10000 (98.86%)


Epoch 4
Train: Loss=0.0152, Batch_id=1874, Accuracy=99.09: 100%|██████████| 1875/1875 [00:34<00:00, 53.79it/s]
Test set: Average loss: 0.0246, Accuracy: 9924/10000 (99.24%)


Epoch 5
Train: Loss=0.0268, Batch_id=1874, Accuracy=99.19: 100%|██████████| 1875/1875 [00:34<00:00, 54.90it/s]
Test set: Average loss: 0.0222, Accuracy: 9933/10000 (99.33%)


Epoch 6
Train: Loss=0.0170, Batch_id=1874, Accuracy=99.17: 100%|██████████| 1875/1875 [00:35<00:00, 52.27it/s]
Test set: Average loss: 0.0201, Accuracy: 9941/10000 (99.41%)


Epoch 7
Train: Loss=0.0076, Batch_id=1874, Accuracy=99.27: 100%|██████████| 1875/1875 [00:34<00:00, 53.94it/s]
Test set: Average loss: 0.0225, Accuracy: 9930/10000 (99.30%)


Epoch 8
Train: Loss=0.0512, Batch_id=1874, Accuracy=99.36: 100%|██████████| 1875/1875 [00:35<00:00, 52.36it/s]
Test set: Average loss: 0.0195, Accuracy: 9934/10000 (99.34%)


Epoch 9
Train: Loss=0.0092, Batch_id=1874, Accuracy=99.40: 100%|██████████| 1875/1875 [00:37<00:00, 50.26it/s]
Test set: Average loss: 0.0192, Accuracy: 9942/10000 (99.42%)


Epoch 10
Train: Loss=0.0084, Batch_id=1874, Accuracy=99.50: 100%|██████████| 1875/1875 [00:34<00:00, 54.20it/s]
Test set: Average loss: 0.0195, Accuracy: 9939/10000 (99.39%)


Epoch 11
Train: Loss=0.0511, Batch_id=1874, Accuracy=99.53: 100%|██████████| 1875/1875 [00:36<00:00, 51.96it/s]
Test set: Average loss: 0.0185, Accuracy: 9945/10000 (99.45%)


Epoch 12
Train: Loss=0.1626, Batch_id=1874, Accuracy=99.54: 100%|██████████| 1875/1875 [00:35<00:00, 53.12it/s]
Test set: Average loss: 0.0175, Accuracy: 9947/10000 (99.47%)


Epoch 13
Train: Loss=0.0007, Batch_id=1874, Accuracy=99.52: 100%|██████████| 1875/1875 [00:34<00:00, 53.89it/s]
Test set: Average loss: 0.0181, Accuracy: 9941/10000 (99.41%)


Epoch 14
Train: Loss=0.0062, Batch_id=1874, Accuracy=99.54: 100%|██████████| 1875/1875 [00:36<00:00, 51.81it/s]
Test set: Average loss: 0.0187, Accuracy: 9941/10000 (99.41%)


Epoch 15
Train: Loss=0.0143, Batch_id=1874, Accuracy=99.56: 100%|██████████| 1875/1875 [00:36<00:00, 50.88it/s]
Test set: Average loss: 0.0183, Accuracy: 9943/10000 (99.43%)
```

<br>

## Train and Test Metrics (Last Iteration)

Last iteration metric plots
![](<../Files/Test and Train Metrics.png>)

<br>

## Receptive Field Calculation (Last Iteration)

The below calculations can be found in this [Excel file](<ERA V1 - Viraj - Assignment 07 - RF Calculation.xlsx>)

![](<../Files/Receptive Field Calculations.png>)

<br>
