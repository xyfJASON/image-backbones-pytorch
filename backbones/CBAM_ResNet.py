"""
Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon.
CBAM: Convolutional Block Attention Module
https://arxiv.org/abs/1807.06521
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CBAM_ResNet', 'cbam_resnet18', 'cbam_resnet34', 'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def conv3x3(in_channels: int, out_channels: int, stride: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def imagenet_first_block():
    """ 3x224x224 -> 64x112x112 -> 64x64x64 """
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


def cifar10_first_block():
    """ 3x32x32 -> 64x32x32 """
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )


class ChannelAttention(nn.Module):
    def __init__(self, C: int, r: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(C, C // r, kernel_size=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(C // r, C, kernel_size=(1, 1)))

    def forward(self, X: torch.Tensor):
        N, C, _, _ = X.shape
        avgpool = self.mlp(self.avgpool(X))
        maxpool = self.mlp(self.maxpool(X))
        weight = torch.sigmoid(avgpool + maxpool)
        return weight * X


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, X: torch.Tensor):
        avgpool = torch.mean(X, dim=1, keepdim=True)
        maxpool, _ = torch.max(X, dim=1, keepdim=True)
        weight = torch.sigmoid(self.conv(torch.cat([avgpool, maxpool], dim=1)))
        return weight * X


class CBAM_BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduce: bool):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, 2 if reduce else 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, 2 if reduce else 1),
                nn.BatchNorm2d(out_channels),
            )
        self.ca = ChannelAttention(out_channels, 16)
        self.sa = SpatialAttention()

    def forward(self, X: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(X)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.sa(self.ca(out)) + self.shortcut(X)
        out = F.relu(out, inplace=True)
        return out


class CBAM_BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, reduce: bool):
        super().__init__()
        self.conv1 = conv1x1(in_channels, mid_channels, 2 if reduce else 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, 1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, 2 if reduce else 1),
                nn.BatchNorm2d(out_channels),
            )
        self.ca = ChannelAttention(out_channels, 16)
        self.sa = SpatialAttention()

    def forward(self, X: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(X)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.sa(self.ca(out)) + self.shortcut(X)
        out = F.relu(out, inplace=True)
        return out


class CBAM_ResNet(nn.Module):
    def __init__(self, block_type: str, n_blocks: List[int], first_block: nn.Module, n_classes: int):
        assert block_type == 'basic' or block_type == 'bottleneck'
        super().__init__()
        self.first_block = first_block
        if block_type == 'basic':
            self.conv2_x = self._make_layer(CBAM_BasicBlock, n_blocks[0], [64, 64], reduce=False)
            self.conv3_x = self._make_layer(CBAM_BasicBlock, n_blocks[1], [64, 128], reduce=True)
            self.conv4_x = self._make_layer(CBAM_BasicBlock, n_blocks[2], [128, 256], reduce=True)
            self.conv5_x = self._make_layer(CBAM_BasicBlock, n_blocks[3], [256, 512], reduce=True)
        else:
            self.conv2_x = self._make_layer(CBAM_BottleneckBlock, n_blocks[0], [64, 64, 256], reduce=False)
            self.conv3_x = self._make_layer(CBAM_BottleneckBlock, n_blocks[1], [256, 128, 512], reduce=True)
            self.conv4_x = self._make_layer(CBAM_BottleneckBlock, n_blocks[2], [512, 256, 1024], reduce=True)
            self.conv5_x = self._make_layer(CBAM_BottleneckBlock, n_blocks[3], [1024, 512, 2048], reduce=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 if block_type == 'basic' else 2048, n_classes)
        self.apply(weights_init)

    @staticmethod
    def _make_layer(CBAM_ResidualBlock, n_block: int, channels: List[int], reduce: bool):
        layers = []
        for _ in range(n_block):
            layers.append(CBAM_ResidualBlock(*channels, reduce=reduce))
            channels[0] = channels[-1]
            reduce = False
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        X = self.first_block(X)
        X = self.conv2_x(X)
        X = self.conv3_x(X)
        X = self.conv4_x(X)
        X = self.conv5_x(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X


def cbam_resnet18(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = CBAM_ResNet('basic', [2, 2, 2, 2], first_block, n_classes=n_classes)
    return model


def cbam_resnet34(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = CBAM_ResNet('basic', [3, 4, 6, 3], first_block, n_classes=n_classes)
    return model


def cbam_resnet50(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = CBAM_ResNet('bottleneck', [3, 4, 6, 3], first_block, n_classes=n_classes)
    return model


def cbam_resnet101(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = CBAM_ResNet('bottleneck', [3, 4, 23, 3], first_block, n_classes=n_classes)
    return model


def cbam_resnet152(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = CBAM_ResNet('bottleneck', [3, 8, 36, 3], first_block, n_classes=n_classes)
    return model


def _test():
    model = cbam_resnet18(n_classes=10)
    X = torch.randn(10, 3, 32, 32)
    out = model(X)
    print(out.shape)
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))
    # from torch.utils.tensorboard import SummaryWriter
    # with SummaryWriter('arch/se_resnet18') as w:
    #     w.add_graph(model, X)


if __name__ == '__main__':
    _test()
