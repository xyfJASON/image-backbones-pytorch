import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T

import backbones
from utils.metrics import SimpleClassificationEvaluator
from utils.general_utils import optimizer_to_device, parse_config


class Trainer:
    def __init__(self, config_path: str):
        self.config, self.device, self.log_root = parse_config(config_path)
        self.train_loader, self.valid_loader, self.n_classes = self._get_data()
        self.model, self.optimizer, self.scheduler, self.criterion = self._prepare_training()
        self.start_epoch, self.best_acc = self.load_ckpt(self.config['resume_path']) if self.config['resume_path'] else (0, 0.0)
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

    def _get_data(self):
        print('==> Getting data...')
        if self.config['dataset'] == 'cifar10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
            test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            train_set = dset.CIFAR10(root=self.config['dataroot'], train=True, transform=train_transforms, download=False)
            test_set = dset.CIFAR10(root=self.config['dataroot'], train=False, transform=test_transforms, download=False)
            n_classes = 10
        elif self.config['dataset'] == 'cifar100':
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
            test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            train_set = dset.CIFAR100(root=self.config['dataroot'], train=True, transform=train_transforms, download=False)
            test_set = dset.CIFAR100(root=self.config['dataroot'], train=False, transform=test_transforms, download=False)
            n_classes = 100
        else:
            raise ValueError(f"Dataset {self.config['dataset']} is not supported now.")
        train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=self.config['batch_size'], num_workers=4, pin_memory=True)
        return train_loader, test_loader, n_classes

    def _prepare_training(self):
        print('==> Preparing training...')
        # =================== DEFINE MODEL =================== #
        if self.config['model'] == 'vgg11':
            model = backbones.vgg11(n_classes=self.n_classes)
        elif self.config['model'] == 'vgg19_bn':
            model = backbones.vgg19_bn(n_classes=self.n_classes)
        elif self.config['model'] == 'resnet18':
            model = backbones.resnet18(n_classes=self.n_classes)
        elif self.config['model'] == 'preactresnet18':
            model = backbones.preactresnet18(n_classes=self.n_classes)
        elif self.config['model'] == 'resnext29_32x4d':
            model = backbones.resnext29_32x4d(n_classes=self.n_classes)
        elif self.config['model'] == 'resnext29_2x64d':
            model = backbones.resnext29_2x64d(n_classes=self.n_classes)
        elif self.config['model'] == 'se_resnet18':
            model = backbones.se_resnet18(n_classes=self.n_classes)
        elif self.config['model'] == 'cbam_resnet18':
            model = backbones.cbam_resnet18(n_classes=self.n_classes)
        elif self.config['model'] == 'mobilenet':
            model = backbones.mobilenet(n_classes=self.n_classes)
        elif self.config['model'] == 'shufflenet_1_0x_g8':
            model = backbones.shufflenet_1_0x_g8(n_classes=self.n_classes)
        elif self.config['model'] == 'vit_tiny':
            model = backbones.vit_tiny(n_classes=self.n_classes, img_size=32, patch_size=4)
        elif self.config['model'] == 'vit_small':
            model = backbones.vit_small(n_classes=self.n_classes, img_size=32, patch_size=4)
        elif self.config['model'] == 'vit_base':
            model = backbones.vit_base(n_classes=self.n_classes, img_size=32, patch_size=4)
        else:
            raise ValueError
        model.to(device=self.device)
        # =================== DEFINE OPTIMIZER =================== #
        if self.config['optimizer']['choice'] == 'sgd':
            cfg = self.config['optimizer']['sgd']
            optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
        elif self.config['optimizer']['choice'] == 'adam':
            cfg = self.config['optimizer']['adam']
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            raise ValueError
        # =================== DEFINE SCHEDULER =================== #
        if self.config['scheduler']['choice'] == 'cosineannealinglr':
            cfg = self.config['scheduler']['cosineannealinglr']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=cfg['eta_min'])
        elif self.config['scheduler']['choice'] == 'multisteplr':
            cfg = self.config['scheduler']['multisteplr']
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
        else:
            raise ValueError
        # =================== DEFINE CRITERION =================== #
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, scheduler, criterion

    def load_ckpt(self, resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device=self.device)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_to_device(self.optimizer, self.device)
        self.scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        print(f'Load checkpoint at epoch {start_epoch - 1}.')
        print(f'Best accuracy so far: {best_acc}')
        return start_epoch, best_acc

    def save_ckpt(self, ep):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': ep,
                    'best_acc': self.best_acc},
                   os.path.join(self.log_root, 'ckpt', f'epoch_{ep}.pt'))

    def load_best_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(device=self.device)

    def save_best_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def train(self):
        print('==> Training...')
        for ep in range(self.start_epoch, self.config['epochs']):
            self.train_one_epoch(ep)

            train_acc = self.evaluate_acc(self.train_loader)
            valid_acc = self.evaluate_acc(self.valid_loader)
            print(f'train accuracy: {train_acc}\nvalid accuracy: {valid_acc}')
            self.writer.add_scalar('Acc/train', train_acc, ep)
            self.writer.add_scalar('Acc/valid', valid_acc, ep)

            if valid_acc > self.best_acc:
                self.best_acc = valid_acc
                self.save_best_model(os.path.join(self.log_root, 'best_model.pt'))

            if self.config['save_per_epochs'] and (ep + 1) % self.config['save_per_epochs'] == 0:
                self.save_ckpt(ep)

        self.writer.close()

    def train_one_epoch(self, ep):
        self.model.train()
        with tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120) as pbar:
            for it, (X, y) in enumerate(pbar):
                X = X.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.long)
                scores = self.model(X)
                loss = self.criterion(scores, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Loss/train', loss.item(), it + ep * len(self.train_loader))
                pbar.set_postfix({'loss': loss.item()})
        self.scheduler.step()

    @torch.no_grad()
    def evaluate_acc(self, dataloader=None):
        if dataloader is None:
            dataloader = self.valid_loader
        self.model.eval()
        evaluator = SimpleClassificationEvaluator()
        for X, y in tqdm(dataloader, desc='Evaluating', ncols=120, leave=False, colour='yellow'):
            X = X.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.long)
            scores = self.model(X)
            _, preds = scores.max(dim=1)
            evaluator.update(preds, y)
        return evaluator.Accuracy()
