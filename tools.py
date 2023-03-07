from yacs.config import CfgNode as CN

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import backbones
from utils.scheduler import LRWarmupWrapper


def build_model(cfg: CN):
    if cfg.model.name.lower() == 'vgg11':
        model = backbones.vgg11(
            n_classes=cfg.data.n_classes,
        )
    elif cfg.model.name.lower() == 'vgg19_bn':
        model = backbones.vgg19_bn(
            n_classes=cfg.data.n_classes,
        )
    elif cfg.model.name.lower() == 'resnet18':
        model = backbones.resnet18(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'preactresnet18':
        model = backbones.preactresnet18(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'resnext29_32x4d':
        model = backbones.resnext29_32x4d(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'resnext29_2x64d':
        model = backbones.resnext29_2x64d(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'se_resnet18':
        model = backbones.se_resnet18(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'cbam_resnet18':
        model = backbones.cbam_resnet18(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'mobilenet':
        model = backbones.mobilenet(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'shufflenet_1_0x_g8':
        model = backbones.shufflenet_1_0x_g8(
            n_classes=cfg.data.n_classes,
            first_block=cfg.model.first_block,
        )
    elif cfg.model.name.lower() == 'vit_tiny':
        model = backbones.vit_tiny(
            n_classes=cfg.data.n_classes,
            img_size=cfg.data.img_size,
            patch_size=cfg.model.patch_size,
        )
    elif cfg.model.name.lower() == 'vit_small':
        model = backbones.vit_small(
            n_classes=cfg.data.n_classes,
            img_size=cfg.data.img_size,
            patch_size=cfg.model.patch_size,
        )
    elif cfg.model.name.lower() == 'vit_base':
        model = backbones.vit_base(
            n_classes=cfg.data.n_classes,
            img_size=cfg.data.img_size,
            patch_size=cfg.model.patch_size,
        )
    else:
        raise ValueError
    return model


def build_optimizer(params, cfg: CN):
    cfg = cfg.train.optim
    if cfg.type.lower() == 'sgd':
        optimizer = optim.SGD(
            params=params,
            lr=cfg.lr,
            momentum=getattr(cfg, 'momentum', 0),
            weight_decay=getattr(cfg, 'weight_decay', 0),
            nesterov=getattr(cfg, 'nesterov', False),
        )
    elif cfg.type.lower() == 'adam':
        optimizer = optim.Adam(
            params=params,
            lr=cfg.lr,
            betas=getattr(cfg, 'betas', (0.9, 0.999)),
            weight_decay=getattr(cfg, 'weight_decay', 0),
        )
    else:
        raise ValueError(f"Optimizer {cfg.type} is not supported.")
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: CN):
    cfg = cfg.train.sched
    if cfg.type.lower() == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.t_max,
            eta_min=cfg.eta_min,
        )
    elif cfg.type.lower() == 'multisteplr':
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    else:
        raise ValueError(f"Scheduler {cfg.type} is not supported.")

    if getattr(cfg, 'warmup_steps', 0) > 0:
        scheduler = LRWarmupWrapper(
            torch_scheduler=scheduler,
            warmup_steps=cfg.warmup_steps,
            warmup_factor=cfg.warmup_factor,
        )
    return scheduler
