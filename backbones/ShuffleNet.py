"""
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
https://arxiv.org/abs/1707.01083
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ShuffleNet', 'shufflenet_1_0x_g2', 'shufflenet_1_0x_g3', 'shufflenet_1_0x_g4', 'shufflenet_1_0x_g8']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def channel_shuffle(X: torch.Tensor, n_groups: int):
    N, C, H, W = X.shape; c = C // n_groups
    return torch.transpose(X.view(N, n_groups, c, H, W), 1, 2).contiguous().view(N, C, H, W)


class BasicUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_groups: int, reduce: bool):
        super().__init__()
        self.n_groups = n_groups
        self.reduce = reduce
        mid_channels = out_channels // 4
        if reduce:
            out_channels -= in_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), groups=1 if in_channels == 24 else n_groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(2, 2) if reduce else (1, 1), padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1), groups=n_groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, X: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(X)), inplace=True)
        out = channel_shuffle(out, self.n_groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out = torch.cat((out, self.avgpool(X)), dim=1) if self.reduce else out + X
        out = F.relu(out, inplace=True)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, n_blocks: List[int], hidden_channels: int, n_groups: int, first_block: nn.Module, n_classes: int):
        super().__init__()
        self.first_block = first_block
        self.stage2 = self._make_layer(24, hidden_channels, n_blocks[0], n_groups)
        self.stage3 = self._make_layer(hidden_channels, hidden_channels*2, n_blocks[1], n_groups)
        self.stage4 = self._make_layer(hidden_channels*2, hidden_channels*4, n_blocks[2], n_groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_channels*4, n_classes)
        self.apply(weights_init)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, n_block: int, n_groups: int):
        layers = [BasicUnit(in_channels, out_channels, n_groups, True)]
        layers.extend([BasicUnit(out_channels, out_channels, n_groups, False) for _ in range(n_block-1)])
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        X = self.first_block(X)
        X = self.stage2(X)
        X = self.stage3(X)
        X = self.stage4(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        X = self.fc(X)
        return X


def imagenet_first_block():
    return nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )


def cifar10_first_block():
    return nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(inplace=True),
    )


def shufflenet_1_0x_g8(n_classes: int, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ShuffleNet(n_blocks=[4, 8, 4], hidden_channels=384, n_groups=8, first_block=first_block, n_classes=n_classes)
    return model


def shufflenet_1_0x_g4(n_classes: int, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ShuffleNet(n_blocks=[4, 8, 4], hidden_channels=272, n_groups=4, first_block=first_block, n_classes=n_classes)
    return model


def shufflenet_1_0x_g3(n_classes: int, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ShuffleNet(n_blocks=[4, 8, 4], hidden_channels=240, n_groups=3, first_block=first_block, n_classes=n_classes)
    return model


def shufflenet_1_0x_g2(n_classes: int, first_block: str = 'cifar10'):
    assert first_block in ['cifar10', 'imagenet']
    first_block = cifar10_first_block() if first_block == 'cifar10' else imagenet_first_block()
    model = ShuffleNet(n_blocks=[4, 8, 4], hidden_channels=200, n_groups=2, first_block=first_block, n_classes=n_classes)
    return model


def _test():
    model = shufflenet_1_0x_g8(n_classes=10)
    X = torch.randn(10, 3, 32, 32)
    out = model(X)
    print(out.shape)
    print(sum(param.numel() for param in model.parameters() if param.requires_grad))
    # from torch.utils.tensorboard import SummaryWriter
    # with SummaryWriter('arch/shufflenet_1_0x_g8') as w:
    #     w.add_graph(model, X)


if __name__ == '__main__':
    _test()
