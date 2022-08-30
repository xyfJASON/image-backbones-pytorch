import os
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import backbones
from utils.data import build_dataset, build_dataloader
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.metrics import AverageMeter, accuracy
from utils.train_utils import reduce_tensor, set_device, create_log_directory, save_ckpt, load_ckpt, save_model


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
            self.log_root = create_log_directory(self.config, config_path)

        # ====================================================== #
        # TENSORBOARD
        # ====================================================== #
        if self.is_master:
            os.makedirs(os.path.join(self.log_root, 'tensorboard'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

        # ====================================================== #
        # DATA
        # ====================================================== #
        train_dataset, self.n_classes = build_dataset(self.config['dataset'], dataroot=self.config['dataroot'], img_size=32, split='train')
        valid_dataset, self.n_classes = build_dataset(self.config['dataset'], dataroot=self.config['dataroot'], img_size=32, split='test')
        self.train_loader = build_dataloader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, is_ddp=self.is_ddp)
        self.valid_loader = build_dataloader(valid_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, is_ddp=self.is_ddp)

        # ====================================================== #
        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        # ====================================================== #
        self.model = self.build_model()
        self.optimizer = build_optimizer(self.model.parameters(), self.config['optimizer'])
        self.scheduler = build_scheduler(self.optimizer, self.config['scheduler'])
        # resume
        if self.config['resume']['ckpt_path']:
            self.start_epoch, self.best_acc = load_ckpt(self.config['resume']['ckpt_path'], self.model, self.optimizer, self.scheduler, self.device)
        else:
            self.start_epoch, self.best_acc = 0, 0.0
        # distributed
        if self.is_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # ====================================================== #
        # DEFINE LOSSES
        # ====================================================== #
        self.criterion = nn.CrossEntropyLoss()

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

    def train(self):
        print('==> Training...')
        for ep in range(self.start_epoch, self.config['epochs']):
            if self.is_ddp:
                dist.barrier()
                self.train_loader.sampler.set_epoch(ep)

            self.train_one_epoch(ep)

            if self.is_master:
                if self.config.get('save_freq') and (ep + 1) % self.config['save_freq'] == 0:
                    save_ckpt(os.path.join(self.log_root, 'ckpt', f'epoch_{ep}.pt'), self.model, self.optimizer, self.scheduler, ep, self.best_acc)

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
                    save_model(os.path.join(self.log_root, 'best_model.pt'), self.model)
                self.writer.add_scalar('LR/group0', self.optimizer.param_groups[0]['lr'], ep)

        if self.is_master:
            print(f'Best valid accuracy: {self.best_acc}')
            self.writer.close()

    def train_one_epoch(self, ep):
        self.model.train()

        if self.is_master:
            pbar = tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120)
        else:
            pbar = self.train_loader

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

        if self.is_master:
            pbar.close()

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
