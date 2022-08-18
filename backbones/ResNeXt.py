"""
Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
Aggregated Residual Transformations for Deep Neural Networks
https://arxiv.org/abs/1611.05431
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNeXt', 'resnext29_2x64d', 'resnext29_32x4d', 'resnext50_32x4d', 'resnext101_64x4d', 'resnext101_32x4d']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def conv3x3(in_channels: int, out_channels: int, stride: int, groups: int):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)


def conv1x1(in_channels: int, out_channels: int, stride: int):
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


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


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, cardinality: int, reduce: bool):
        super().__init__()
        assert mid_channels % cardinality == 0
        self.conv1 = conv1x1(in_channels, mid_channels, 2 if reduce else 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, mid_channels, 1, cardinality)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = conv1x1(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if reduce or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, 2 if reduce else 1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, X: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(X)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = F.relu(self.bn3(self.conv3(out)) + self.shortcut(X), inplace=True)
        return out


class ResNeXt(nn.Module):
    def __init__(self, n_blocks: List[int], cfg: Tuple[int, int], first_block: nn.Module, n_classes: int):
        super().__init__()
        self.first_block = first_block
        self.in_channels = 64
        self.cardinality, self.width = cfg
        self.conv2 = self._make_layer(n_blocks[0], reduce=False)
        self.conv3 = self._make_layer(n_blocks[1], reduce=True)
        self.conv4 = self._make_layer(n_blocks[2], reduce=True)
        self.conv5 = nn.Sequential() if len(n_blocks) == 3 else self._make_layer(n_blocks[3], reduce=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.cardinality * self.width, n_classes)
        self.apply(weights_init)

    def _make_layer(self, n_block: int, reduce: bool):
        layers = []
        for _ in range(n_block):
            channels = self.cardinality * self.width
            layers.append(BottleneckBlock(self.in_channels, channels, channels * 2, self.cardinality, reduce=reduce))
            self.in_channels = channels * 2
            reduce = False
        self.width *= 2
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        X = self.first_block(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X


def resnext29_2x64d(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ResNeXt([3, 3, 3], (2, 64), first_block, n_classes=n_classes)
    return model


def resnext29_32x4d(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ResNeXt([3, 3, 3], (32, 4), first_block, n_classes=n_classes)
    return model


def resnext50_32x4d(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ResNeXt([3, 4, 6, 3], (32, 4), first_block, n_classes=n_classes)
    return model


def resnext101_32x4d(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ResNeXt([3, 4, 23, 3], (32, 4), first_block, n_classes=n_classes)
    return model


def resnext101_64x4d(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ResNeXt([3, 4, 23, 3], (64, 4), first_block, n_classes=n_classes)
    return model


def _test():
    model = resnext29_32x4d(n_classes=10)
    X = torch.randn(10, 3, 32, 32)
    out = model(X)
    print(out.shape)
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))
    # from torch.utils.tensorboard import SummaryWriter
    # with SummaryWriter('arch/resnext29_32x4d') as w:
    #     w.add_graph(model, X)


if __name__ == '__main__':
    _test()
