"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Imported from:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic block"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model"""

    # Class variable to print shape
    print_shape = False

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Make layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def print_view(self, x, msg=""):
        """Print shape of the model"""
        if self.print_shape:
            if msg != "":
                print(msg, "\n\t", x.shape, "\n")
            else:
                print(x.shape)

    def forward(self, x):
        """Forward pass"""

        # Input layer
        out = F.relu(self.bn1(self.conv1(x)))
        self.print_view(out, "PrepLayer")

        # Layer 1
        out = self.layer1(out)
        self.print_view(out, "Layer 1")

        # Layer 2
        out = self.layer2(out)
        self.print_view(out, "Layer 2")

        # Layer 3
        out = self.layer3(out)
        self.print_view(out, "Layer 3")

        # Layer 4
        out = self.layer4(out)
        self.print_view(out, "Layer 4")

        # GAP layer
        out = F.avg_pool2d(out, 4)
        self.print_view(out, "Post GAP")

        # Reshape before FC such that it becomes 1D
        out = out.view(out.size(0), -1)
        self.print_view(out, "Reshape before FC")

        # FC Layer
        out = self.linear(out)
        self.print_view(out, "After FC")
        return out


def ResNet18():
    """ResNet18 model"""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    """ResNet34 model"""
    return ResNet(BasicBlock, [3, 4, 6, 3])


def test():
    """Test function for ResNet"""
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
