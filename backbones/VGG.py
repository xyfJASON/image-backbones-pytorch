"""
Karen Simonyan, Andrew Zisserman
Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556
"""

import torch
import torch.nn as nn

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def conv3re(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),  # noqa
        nn.ReLU(inplace=True),
    )


def conv3bnre(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),  # noqa
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class VGG(nn.Module):
    def __init__(self, cfg: list, bn: bool, n_classes: int) -> None:
        super().__init__()
        self.features = self._make_layer(cfg, bn)
        self.flatten = nn.Flatten()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(512, n_classes)
        self.apply(weights_init)

    def _make_layer(self, cfg: list, bn: bool) -> nn.Sequential:  # noqa
        layers = []
        lst = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(conv3bnre(lst, x) if bn else conv3re(lst, x))
                lst = x
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.features(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.classifier(X)
        return X


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


def _test() -> None:
    model = vgg11(n_classes=10)
    X = torch.randn(10, 3, 32, 32)
    out = model(X)
    print(out.shape)
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))
    # from torch.utils.tensorboard import SummaryWriter
    # with SummaryWriter('arch/vgg11') as w:
    #     w.add_graph(model, X)


if __name__ == '__main__':
    _test()
