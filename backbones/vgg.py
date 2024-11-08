"""
Karen Simonyan, Andrew Zisserman
Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def conv3re(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )


def conv3bnre(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class VGG(nn.Module):
    def __init__(self, cfg: List, bn: bool, n_classes: int):
        super().__init__()
        self.features = self._make_layer(cfg, bn)
        self.flatten = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(512, n_classes)
        self.apply(weights_init)

    @staticmethod
    def _make_layer(cfg: List, bn: bool):
        layers = []
        lst = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(conv3bnre(lst, x) if bn else conv3re(lst, x))
                lst = x
        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, bn: bool, n_classes: int) -> VGG:
    model = VGG(cfg=cfgs[arch.replace('_bn', '')], bn=bn, n_classes=n_classes)
    return model


def vgg11(n_classes: int) -> VGG:
    return _vgg('vgg11', False, n_classes)


def vgg13(n_classes: int) -> VGG:
    return _vgg('vgg13', False, n_classes)


def vgg16(n_classes: int) -> VGG:
    return _vgg('vgg16', False, n_classes)


def vgg19(n_classes: int) -> VGG:
    return _vgg('vgg19', False, n_classes)


def vgg11_bn(n_classes: int) -> VGG:
    return _vgg('vgg11_bn', True, n_classes)


def vgg13_bn(n_classes: int) -> VGG:
    return _vgg('vgg13_bn', True, n_classes)


def vgg16_bn(n_classes: int) -> VGG:
    return _vgg('vgg16_bn', True, n_classes)


def vgg19_bn(n_classes: int) -> VGG:
    return _vgg('vgg19_bn', True, n_classes)


def _test_overhead():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.overhead import calc_flops, count_params, calc_inference_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = vgg11(n_classes=10).to(device)
    model = vgg19_bn(n_classes=10).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)

    count_params(model)
    print('=' * 60)
    calc_flops(model, x)
    print('=' * 60)
    calc_inference_time(model, x)


if __name__ == '__main__':
    _test_overhead()
