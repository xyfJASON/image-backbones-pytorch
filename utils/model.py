import backbones


def build_model(name, n_classes, img_size):
    if name == 'vgg11':
        model = backbones.vgg11(n_classes=n_classes)
    elif name == 'vgg19_bn':
        model = backbones.vgg19_bn(n_classes=n_classes)
    elif name == 'resnet18':
        model = backbones.resnet18(n_classes=n_classes)
    elif name == 'preactresnet18':
        model = backbones.preactresnet18(n_classes=n_classes)
    elif name == 'resnext29_32x4d':
        model = backbones.resnext29_32x4d(n_classes=n_classes)
    elif name == 'resnext29_2x64d':
        model = backbones.resnext29_2x64d(n_classes=n_classes)
    elif name == 'se_resnet18':
        model = backbones.se_resnet18(n_classes=n_classes)
    elif name == 'cbam_resnet18':
        model = backbones.cbam_resnet18(n_classes=n_classes)
    elif name == 'mobilenet':
        model = backbones.mobilenet(n_classes=n_classes)
    elif name == 'shufflenet_1_0x_g8':
        model = backbones.shufflenet_1_0x_g8(n_classes=n_classes)
    elif name == 'vit_tiny':
        model = backbones.vit_tiny(n_classes=n_classes, img_size=img_size, patch_size=4)
    elif name == 'vit_small':
        model = backbones.vit_small(n_classes=n_classes, img_size=img_size, patch_size=4)
    elif name == 'vit_base':
        model = backbones.vit_base(n_classes=n_classes, img_size=img_size, patch_size=4)
    else:
        raise ValueError
    return model
