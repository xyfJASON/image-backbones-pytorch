"""
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn

__all__ = ['MobileNet', 'mobilenet']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def DSConvBnRe(in_channels: int, out_channels: int, reduce: bool):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2 if reduce else 1, padding=1, bias=False, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, first_block: nn.Module, n_classes: int):
        super().__init__()
        self.first_block = first_block
        self.conv1 = DSConvBnRe(32, 64, reduce=False)
        self.conv2 = DSConvBnRe(64, 128, reduce=True)
        self.conv3 = DSConvBnRe(128, 128, reduce=False)
        self.conv4 = DSConvBnRe(128, 256, reduce=True)
        self.conv5 = DSConvBnRe(256, 256, reduce=False)
        self.conv6 = DSConvBnRe(256, 512, reduce=True)
        self.conv71 = DSConvBnRe(512, 512, reduce=False)
        self.conv72 = DSConvBnRe(512, 512, reduce=False)
        self.conv73 = DSConvBnRe(512, 512, reduce=False)
        self.conv74 = DSConvBnRe(512, 512, reduce=False)
        self.conv75 = DSConvBnRe(512, 512, reduce=False)
        self.conv8 = DSConvBnRe(512, 1024, reduce=True)
        self.conv9 = DSConvBnRe(1024, 1024, reduce=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, n_classes)
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        X = self.first_block(X)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.conv6(X)
        X = self.conv71(X)
        X = self.conv72(X)
        X = self.conv73(X)
        X = self.conv74(X)
        X = self.conv75(X)
        X = self.conv8(X)
        X = self.conv9(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X


def imagenet_first_block():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


def cifar10_first_block():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
    )


def mobilenet(n_classes, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = MobileNet(first_block, n_classes=n_classes)
    return model


def _test_overhead():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.overhead import calc_flops, count_params, calc_inference_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet(n_classes=10).to(device)
    X = torch.randn(10, 3, 32, 32).to(device)

    count_params(model)
    print('=' * 60)
    calc_flops(model, X)
    print('=' * 60)
    calc_inference_time(model, X)


if __name__ == '__main__':
    _test_overhead()
