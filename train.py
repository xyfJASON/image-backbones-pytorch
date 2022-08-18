import os
import yaml
import shutil
import datetime
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as dset
import torchvision.transforms as T

import backbones
from utils.metrics import AverageMeter, accuracy
from utils.train_utils import optimizer_to_device, reduce_tensor, set_device


class Trainer:
    def __init__(self, config_path: str):
        # ====================================================== #
        # READ CONFIGURATION FILE
        # ====================================================== #
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ====================================================== #
        # SET DEVICE
        # ====================================================== #
        self.device, self.world_size, self.local_rank, self.global_rank = set_device()
        self.is_master = self.world_size <= 1 or self.global_rank == 0
        self.is_ddp = self.world_size > 1
        print('using device:', self.device)

        # ====================================================== #
        # CREATE LOG DIRECTORY
        # ====================================================== #
        if self.is_master:
            if self.config['resume']['log_root'] is not None:
                assert os.path.isdir(self.config['resume']['log_root'])
                self.log_root = self.config['resume']['log_root']
            else:
                self.log_root = os.path.join('runs', datetime.datetime.now().strftime('exp-%Y-%m-%d-%H-%M-%S'))
                os.makedirs(self.log_root)
            print('log directory:', self.log_root)

            if self.config.get('save_freq'):
                os.makedirs(os.path.join(self.log_root, 'ckpt'), exist_ok=True)
            if self.config.get('sample_freq'):
                os.makedirs(os.path.join(self.log_root, 'samples'), exist_ok=True)
            if not os.path.exists(os.path.join(self.log_root, 'config.yml')):
                shutil.copyfile(config_path, os.path.join(self.log_root, 'config.yml'))

        # ====================================================== #
        # TENSORBOARD
        # ====================================================== #
        if self.is_master:
            os.makedirs(os.path.join(self.log_root, 'tensorboard'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        else:
            self.writer = None

        # ====================================================== #
        # DATA
        # ====================================================== #
        self.train_loader, self.valid_loader, self.n_classes = self.get_data()

        # ====================================================== #
        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        # ====================================================== #
        self.model = self.build_model()
        self.optimizer = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(self.optimizer)
        # resume
        if self.config['resume']['ckpt_path']:
            self.start_epoch, self.best_acc = self.load_ckpt(self.config['resume']['ckpt_path'])
        else:
            self.start_epoch, self.best_acc = 0, 0.0
        # distributed
        if self.is_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # ====================================================== #
        # DEFINE LOSSES
        # ====================================================== #
        self.criterion = nn.CrossEntropyLoss()

    def get_data(self):
        print('==> Getting data...')
        if self.config['dataset'] == 'cifar10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
            test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            train_set = dset.CIFAR10(root=self.config['dataroot'], train=True, transform=train_transforms, download=False)
            valid_set = dset.CIFAR10(root=self.config['dataroot'], train=False, transform=test_transforms, download=False)
            n_classes = 10
        elif self.config['dataset'] == 'cifar100':
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            train_transforms = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
            test_transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
            train_set = dset.CIFAR100(root=self.config['dataroot'], train=True, transform=train_transforms, download=False)
            valid_set = dset.CIFAR100(root=self.config['dataroot'], train=False, transform=test_transforms, download=False)
            n_classes = 100
        else:
            raise ValueError(f"Dataset {self.config['dataset']} is not supported now.")

        if self.is_ddp:
            sampler = DistributedSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        if self.is_ddp:
            sampler = DistributedSampler(valid_set, shuffle=False)
            valid_loader = DataLoader(valid_set, batch_size=self.config['batch_size'], sampler=sampler, num_workers=4, pin_memory=True)
        else:
            valid_loader = DataLoader(valid_set, batch_size=self.config['batch_size'], num_workers=4, pin_memory=True)

        return train_loader, valid_loader, n_classes

    def build_model(self):
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
        return model

    def build_optimizer(self, model):
        if self.config['optimizer']['choice'] == 'sgd':
            cfg = self.config['optimizer']['sgd']
            optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
        elif self.config['optimizer']['choice'] == 'adam':
            cfg = self.config['optimizer']['adam']
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            raise ValueError
        return optimizer

    def build_scheduler(self, optimizer):
        if self.config['scheduler']['choice'] == 'cosineannealinglr':
            cfg = self.config['scheduler']['cosineannealinglr']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=cfg['eta_min'])
        elif self.config['scheduler']['choice'] == 'multisteplr':
            cfg = self.config['scheduler']['multisteplr']
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
        else:
            raise ValueError
        return scheduler

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
            if self.is_ddp:
                dist.barrier()
                self.train_loader.sampler.set_epoch(ep)

            self.train_one_epoch(ep)

            if self.is_master:
                if self.config.get('save_freq') and (ep + 1) % self.config['save_freq'] == 0:
                    self.save_ckpt(ep)

            train_acc, train_loss = self.evaluate(self.train_loader)
            valid_acc, valid_loss = self.evaluate(self.valid_loader)
            if self.is_master:
                print(f'train accuracy: {train_acc}')
                print(f'valid accuracy: {valid_acc}')
                self.writer.add_scalar('Train/acc', train_acc, ep)
                self.writer.add_scalar('Valid/acc', valid_acc, ep)
                self.writer.add_scalar('Valid/loss', valid_loss, ep)
                if valid_acc > self.best_acc:
                    self.best_acc = valid_acc
                    self.save_best_model(os.path.join(self.log_root, 'best_model.pt'))

        if self.is_master:
            print(f'Best valid accuracy: {self.best_acc}')
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

                if self.is_ddp:
                    loss = reduce_tensor(loss.detach(), self.world_size)
                if self.is_master:
                    self.writer.add_scalar('Train/loss', loss.item(), it + ep * len(self.train_loader))
                pbar.set_postfix({'loss': loss.item()})
        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.valid_loader
        self.model.eval()

        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        for X, y in tqdm(dataloader, desc='Evaluating', ncols=120, leave=False, colour='yellow'):
            X = X.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.long)
            scores = self.model(X)

            acc = accuracy(scores, y)[0]
            loss = self.criterion(scores, y)

            if self.is_ddp:
                acc = reduce_tensor(acc, self.world_size)
                loss = reduce_tensor(loss, self.world_size)

            loss_meter.update(loss.item(), X.shape[0])
            acc_meter.update(acc.item(), X.shape[0])

        return acc_meter.avg, loss_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    args = parser.parse_args()

    trainer = Trainer(args.config_path)
    trainer.train()


if __name__ == '__main__':
    main()
