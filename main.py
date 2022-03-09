import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T

import backbones
from trainer import Trainer
from utils.general_utils import parse_config


if __name__ == '__main__':
    config, device, log_root = parse_config('./config.yml')

    # =================== LOAD DATA =================== #
    print('==> Getting data...')
    if config['dataset'] == 'cifar10':
        mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_set = dset.CIFAR10(root=config['dataroot'], train=True, transform=train_transforms, download=False)
        test_set = dset.CIFAR10(root=config['dataroot'], train=False, transform=test_transforms, download=False)
        n_classes = 10
    elif config['dataset'] == 'cifar100':
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_set = dset.CIFAR100(root=config['dataroot'], train=True, transform=train_transforms, download=False)
        test_set = dset.CIFAR100(root=config['dataroot'], train=False, transform=test_transforms, download=False)
        n_classes = 100
    else:
        raise ValueError(f"Dataset {config['dataset']} is not supported now.")
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], num_workers=4, pin_memory=True)

    # =================== DEFINE MODEL =================== #
    print('==> Preparing training...')
    if config['model'] == 'vgg11':
        model = backbones.vgg11(n_classes=n_classes)
    elif config['model'] == 'vgg19_bn':
        model = backbones.vgg19_bn(n_classes=n_classes)
    elif config['model'] == 'resnet18':
        model = backbones.resnet18(n_classes=n_classes)
    elif config['model'] == 'preactresnet18':
        model = backbones.preactresnet18(n_classes=n_classes)
    elif config['model'] == 'mobilenet':
        model = backbones.mobilenet(n_classes=n_classes)
    elif config['model'] == 'shufflenet_1_0x_g8':
        model = backbones.shufflenet_1_0x_g8(n_classes=n_classes)
    else:
        raise ValueError
    model.to(device=device)

    # =================== DEFINE OPTIMIZER =================== #
    if config['optimizer']['choice'] == 'sgd':
        cfg = config['optimizer']['sgd']
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    elif config['optimizer']['choice'] == 'adam':
        cfg = config['optimizer']['adam']
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    else:
        raise ValueError

    # =================== DEFINE SCHEDULER =================== #
    if config['scheduler']['choice'] == 'cosineannealinglr':
        cfg = config['scheduler']['cosineannealinglr']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=cfg['eta_min'])
    elif config['scheduler']['choice'] == 'multisteplr':
        cfg = config['scheduler']['multisteplr']
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    else:
        raise ValueError

    # =================== DEFINE CRITERION =================== #
    criterion = nn.CrossEntropyLoss()

    # =================== START TRAINING =================== #
    print('==> Training...')
    trainer = Trainer(device, model, optimizer, scheduler, criterion, train_loader, test_loader,
                      config['epochs'], n_classes, log_root, config['save_per_epochs'], config['resume_path'])
    trainer.train()

    # =================== FINAL TEST =================== #
    print('==> Testing...')
    model.load_state_dict(torch.load(os.path.join(log_root, 'best_model.pt'), map_location=device))
    valid_acc = trainer.evaluate_acc(test_loader)
    print(f'valid accuracy: {valid_acc}')
