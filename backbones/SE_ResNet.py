"""
Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
Squeeze-and-Excitation Networks
https://arxiv.org/abs/1709.01507
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SE_ResNet', 'se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152']


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
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


def cifar10_first_block():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )


class SEBlock(nn.Module):
    def __init__(self, C: int, r: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(C, C // r)
        self.fc2 = nn.Linear(C // r, C)

    def forward(self, X: torch.Tensor):
        N, C, _, _ = X.shape
        weight = self.avgpool(X).reshape(N, C)
        weight = F.relu(self.fc1(weight), inplace=True)
        weight = torch.sigmoid(self.fc2(weight))
        return weight.reshape(N, C, 1, 1) * X


class SE_BasicBlock(nn.Module):
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
        self.se = SEBlock(out_channels, 16)

    def forward(self, X: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(X)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out) + self.shortcut(X)
        out = F.relu(out, inplace=True)
        return out


class SE_BottleneckBlock(nn.Module):
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
        self.se = SEBlock(out_channels, 16)

    def forward(self, X: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(X)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out) + self.shortcut(X)
        out = F.relu(out, inplace=True)
        return out


class SE_ResNet(nn.Module):
    def __init__(self, block_type: str, n_blocks: List[int], first_block: nn.Module, n_classes: int):
        assert block_type == 'basic' or block_type == 'bottleneck'
        super().__init__()
        self.first_block = first_block
        if block_type == 'basic':
            self.conv2_x = self._make_layer(SE_BasicBlock, n_blocks[0], [64, 64], reduce=False)
            self.conv3_x = self._make_layer(SE_BasicBlock, n_blocks[1], [64, 128], reduce=True)
            self.conv4_x = self._make_layer(SE_BasicBlock, n_blocks[2], [128, 256], reduce=True)
            self.conv5_x = self._make_layer(SE_BasicBlock, n_blocks[3], [256, 512], reduce=True)
        else:
            self.conv2_x = self._make_layer(SE_BottleneckBlock, n_blocks[0], [64, 64, 256], reduce=False)
            self.conv3_x = self._make_layer(SE_BottleneckBlock, n_blocks[1], [256, 128, 512], reduce=True)
            self.conv4_x = self._make_layer(SE_BottleneckBlock, n_blocks[2], [512, 256, 1024], reduce=True)
            self.conv5_x = self._make_layer(SE_BottleneckBlock, n_blocks[3], [1024, 512, 2048], reduce=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 if block_type == 'basic' else 2048, n_classes)
        self.apply(weights_init)

    @staticmethod
    def _make_layer(SE_ResidualBlock, n_block: int, channels: List[int], reduce: bool):
        layers = []
        for _ in range(n_block):
            layers.append(SE_ResidualBlock(*channels, reduce=reduce))
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


def se_resnet18(n_classes, first_block: str = 'cifar10') -> SE_ResNet:
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = SE_ResNet('basic', [2, 2, 2, 2], first_block, n_classes=n_classes)
    return model


def se_resnet34(n_classes, first_block: str = 'cifar10') -> SE_ResNet:
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = SE_ResNet('basic', [3, 4, 6, 3], first_block, n_classes=n_classes)
    return model


def se_resnet50(n_classes, first_block: str = 'cifar10') -> SE_ResNet:
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = SE_ResNet('bottleneck', [3, 4, 6, 3], first_block, n_classes=n_classes)
    return model


def se_resnet101(n_classes, first_block: str = 'cifar10') -> SE_ResNet:
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = SE_ResNet('bottleneck', [3, 4, 23, 3], first_block, n_classes=n_classes)
    return model


def se_resnet152(n_classes, first_block: str = 'cifar10') -> SE_ResNet:
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = SE_ResNet('bottleneck', [3, 8, 36, 3], first_block, n_classes=n_classes)
    return model


def _test_overhead():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.overhead import calc_flops, count_params, calc_inference_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = se_resnet18(n_classes=10).to(device)
    X = torch.randn(10, 3, 32, 32).to(device)

    count_params(model)
    print('=' * 60)
    calc_flops(model, X)
    print('=' * 60)
    calc_inference_time(model, X)


if __name__ == '__main__':
    _test_overhead()
