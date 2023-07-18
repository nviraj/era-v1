"""Module to define the model."""

import torch.nn as nn
import torch.nn.functional as F

############# Assignment 10 Model #############


# This is for Assignment 10
class CustomResNet(nn.Module):
    """This defines the structure of the NN."""

    # Class variable to print shape
    print_shape = False
    # Default dropout value
    dropout_value = 0.05

    def __init__(self):
        super().__init__()

        #  Model Notes
        # Stride = 2 implemented in last layer of block 2
        # Depthwise separable convolution implemented in block 3
        # Dilated convolution implemented in block 4
        # Global Average Pooling implemented after block 4
        # Output block has fully connected layers

        self.block1 = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(8),
            # Layer 2
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(8),
            # Layer 3
            nn.Conv2d(
                in_channels=8,
                out_channels=12,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(12),
        )

        self.block2 = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=12,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(16),
            # Layer 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32),
            # Layer 3
            nn.Conv2d(
                in_channels=32,
                out_channels=24,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(24),
        )

        self.block3 = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=24,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32),
            ##################### Depthwise Convolution #####################
            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=(3, 3),
                groups=32,
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(128),
            # Layer 3
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(32),
        )

        self.block4 = nn.Sequential(
            ##################### Dilated Convolution #####################
            # Layer 1
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=2,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(64),
            # Layer 2
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(96),
            # Layer 3
            nn.Conv2d(
                in_channels=96,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.BatchNorm2d(64),
        )

        ##################### GAP #####################
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1))

        ##################### Fully Connected Layer #####################
        self.output_block = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))

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
        x = self.gap(x)
        self.print_view(x)
        # Flatten the layer
        x = x.view((x.shape[0], -1))
        self.print_view(x)
        x = self.output_block(x)
        self.print_view(x)
        x = F.log_softmax(x, dim=1)

        return x
