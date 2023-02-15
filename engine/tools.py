import os
from yacs.config import CfgNode as CN

import torch.optim as optim
from torch.utils.data import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import backbones
from utils.logger import get_logger
from utils.scheduler import LRWarmupWrapper


def build_model(cfg: CN):
    if cfg.MODEL.NAME.lower() == 'vgg11':
        model = backbones.vgg11(
            n_classes=cfg.DATA.N_CLASSES,
        )
    elif cfg.MODEL.NAME.lower() == 'vgg19_bn':
        model = backbones.vgg19_bn(
            n_classes=cfg.DATA.N_CLASSES,
        )
    elif cfg.MODEL.NAME.lower() == 'resnet18':
        model = backbones.resnet18(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'preactresnet18':
        model = backbones.preactresnet18(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'resnext29_32x4d':
        model = backbones.resnext29_32x4d(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'resnext29_2x64d':
        model = backbones.resnext29_2x64d(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'se_resnet18':
        model = backbones.se_resnet18(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'cbam_resnet18':
        model = backbones.cbam_resnet18(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'mobilenet':
        model = backbones.mobilenet(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'shufflenet_1_0x_g8':
        model = backbones.shufflenet_1_0x_g8(
            n_classes=cfg.DATA.N_CLASSES,
            first_block=cfg.MODEL.FIRST_BLOCK,
        )
    elif cfg.MODEL.NAME.lower() == 'vit_tiny':
        model = backbones.vit_tiny(
            n_classes=cfg.DATA.N_CLASSES,
            img_size=cfg.DATA.IMG_SIZE,
            patch_size=cfg.MODEL.PATCH_SIZE,
        )
    elif cfg.MODEL.NAME.lower() == 'vit_small':
        model = backbones.vit_small(
            n_classes=cfg.DATA.N_CLASSES,
            img_size=cfg.DATA.IMG_SIZE,
            patch_size=cfg.MODEL.PATCH_SIZE,
        )
    elif cfg.MODEL.NAME.lower() == 'vit_base':
        model = backbones.vit_base(
            n_classes=cfg.DATA.N_CLASSES,
            img_size=cfg.DATA.IMG_SIZE,
            patch_size=cfg.MODEL.PATCH_SIZE,
        )
    else:
        raise ValueError
    return model


def build_optimizer(params, cfg: CN):
    cfg = cfg.TRAIN.OPTIM
    if cfg.TYPE == 'SGD':
        optimizer = optim.SGD(
            params=params,
            lr=cfg.LR,
            momentum=getattr(cfg, 'MOMENTUM', 0),
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0),
            nesterov=getattr(cfg, 'NESTEROV', False),
        )
    elif cfg.TYPE == 'Adam':
        optimizer = optim.Adam(
            params=params,
            lr=cfg.LR,
            betas=getattr(cfg, 'BETAS', (0.9, 0.999)),
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0),
        )
    else:
        raise ValueError(f"Optimizer {cfg.TYPE} is not supported.")
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, cfg: CN):
    cfg = cfg.TRAIN.SCHED
    if cfg.TYPE == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.T_MAX,
            eta_min=cfg.ETA_MIN,
        )
    elif cfg.TYPE == 'MultiStepLR':
        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.MILESTONES,
            gamma=cfg.GAMMA,
        )
    else:
        raise ValueError(f"Scheduler {cfg.TYPE} is not supported.")

    if hasattr(cfg, 'WARMUP_STEPS') and cfg.WARMUP_STEPS > 0:
        scheduler = LRWarmupWrapper(
            torch_scheduler=scheduler,
            warmup_steps=cfg.WARMUP_STEPS,
            warmup_factor=cfg.WARMUP_FACTOR,
        )
    return scheduler


def build_dataset(cfg, split, transforms=None, subset_ids=None, strict_valid_test=False):
    """
    Args:
        cfg: CfgNode
        split: 'train', 'valid', 'test', 'all'
        transforms: if None, will use default transforms
        subset_ids: select a subset of the full dataset
        strict_valid_test: replace validation split with test split (or vice versa)
                           if the dataset doesn't have a validation / test split
    """
    split = _check_split(cfg.DATA.NAME, split, strict_valid_test)

    if cfg.DATA.NAME.lower() in ['cifar10', 'cifar-10']:
        from datasets.CIFAR10 import CIFAR10, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(cfg.DATA.IMG_SIZE, split)
        dataset = CIFAR10(root=cfg.DATA.DATAROOT, train=(split == 'train'), transform=transforms)

    elif cfg.DATA.NAME.lower() in ['cifar100', 'cifar-100']:
        from datasets.CIFAR100 import CIFAR100, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(cfg.DATA.IMG_SIZE, split)
        dataset = CIFAR100(root=cfg.DATA.DATAROOT, train=(split == 'train'), transform=transforms)

    else:
        raise ValueError(f"Dataset {cfg.DATA.NAME} is not supported.")

    if subset_ids is not None and len(subset_ids) < len(dataset):
        dataset = Subset(dataset, subset_ids)
    return dataset


def _check_split(name, split, strict_valid_test):
    available_split = {
        'cifar10': ['train', 'test'],
        'cifar-10': ['train', 'test'],
        'cifar100': ['train', 'test'],
        'cifar-100': ['train', 'test'],
    }
    assert split in ['train', 'valid', 'test', 'all']
    if split in ['train', 'all'] or strict_valid_test:
        assert split in available_split[name.lower()], f"Dataset {name} doesn't have split: {split}"
    elif split not in available_split[name.lower()]:
        replace_split = 'test' if split == 'valid' else 'valid'
        assert replace_split in available_split[name.lower()], f"Dataset {name} doesn't have split: {split}"
        logger = get_logger()
        logger.warning(f'Replace split `{split}` with split `{replace_split}`')
        split = replace_split
    return split


def find_resume_checkpoint(exp_dir, resume):
    """ Checkpoints are named after 'stepxxxxxx.pt' """
    if os.path.isfile(resume):
        ckpt_path = resume
    elif resume == 'best':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'best.pt')
    elif resume == 'latest':
        d = dict()
        for filename in os.listdir(os.path.join(exp_dir, 'ckpt')):
            name, ext = os.path.splitext(filename)
            if ext == '.pt' and name[:4] == 'step':
                d.update({int(name[4:]): filename})
        ckpt_path = os.path.join(exp_dir, 'ckpt', d[sorted(d)[-1]])
    else:
        raise ValueError(f'resume option {resume} is invalid')
    assert os.path.isfile(ckpt_path), f'{ckpt_path} is not a .pt file'
    return ckpt_path
