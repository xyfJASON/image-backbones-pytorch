from omegaconf import DictConfig

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import backbones


def build_model(conf: DictConfig):
    if conf.model.name.lower() == 'vgg11':
        model = backbones.vgg11(
            n_classes=conf.data.n_classes,
        )
    elif conf.model.name.lower() == 'vgg19_bn':
        model = backbones.vgg19_bn(
            n_classes=conf.data.n_classes,
        )
    elif conf.model.name.lower() == 'resnet18':
        model = backbones.resnet18(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'preactresnet18':
        model = backbones.preactresnet18(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'resnext29_32x4d':
        model = backbones.resnext29_32x4d(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'resnext29_2x64d':
        model = backbones.resnext29_2x64d(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'se_resnet18':
        model = backbones.se_resnet18(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'cbam_resnet18':
        model = backbones.cbam_resnet18(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'mobilenet':
        model = backbones.mobilenet(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'shufflenet_1_0x_g8':
        model = backbones.shufflenet_1_0x_g8(
            n_classes=conf.data.n_classes,
            first_block=conf.model.first_block,
        )
    elif conf.model.name.lower() == 'vit_tiny':
        model = backbones.vit_tiny(
            n_classes=conf.data.n_classes,
            img_size=conf.data.img_size,
            patch_size=conf.model.patch_size,
            pdrop=getattr(conf.model, 'pdrop', 0.1),
        )
    elif conf.model.name.lower() == 'vit_small':
        model = backbones.vit_small(
            n_classes=conf.data.n_classes,
            img_size=conf.data.img_size,
            patch_size=conf.model.patch_size,
            pdrop=getattr(conf.model, 'pdrop', 0.1),
        )
    elif conf.model.name.lower() == 'vit_base':
        model = backbones.vit_base(
            n_classes=conf.data.n_classes,
            img_size=conf.data.img_size,
            patch_size=conf.model.patch_size,
            pdrop=getattr(conf.model, 'pdrop', 0.1),
        )
    else:
        raise ValueError(f'Model {conf.model.name} is not supported.')
    return model


def build_optimizer(params, conf: DictConfig):
    conf = conf.train.optim
    if conf.type.lower() == 'sgd':
        optimizer = optim.SGD(
            params=params,
            lr=conf.lr,
            momentum=getattr(conf, 'momentum', 0),
            weight_decay=getattr(conf, 'weight_decay', 0),
            nesterov=getattr(conf, 'nesterov', False),
        )
    elif conf.type.lower() == 'adam':
        optimizer = optim.Adam(
            params=params,
            lr=conf.lr,
            betas=getattr(conf, 'betas', (0.9, 0.999)),
            weight_decay=getattr(conf, 'weight_decay', 0),
        )
    else:
        raise ValueError(f'Optimizer {conf.type} is not supported.')
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, conf: DictConfig):
    conf = conf.train.sched
    if conf.type.lower() == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=conf.t_max,
            eta_min=conf.eta_min,
        )
    elif conf.type.lower() == 'multisteplr':
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=conf.milestones,
            gamma=conf.gamma,
        )
    else:
        raise ValueError(f'Scheduler {conf.type} is not supported.')

    return scheduler
