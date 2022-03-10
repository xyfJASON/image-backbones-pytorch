import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import SimpleClassificationEvaluator
from utils.general_utils import optimizer_to_device


class Trainer:
    def __init__(self, device, model, optimizer, scheduler, criterion, train_loader, valid_loader,
                 epochs, n_classes, log_root, save_per_epochs=None, resume_path=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.n_classes = n_classes
        self.log_root = log_root
        self.save_per_epochs = save_per_epochs

        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.start_epoch, self.best_acc = self.load_ckpt(resume_path) if resume_path else (0, 0.0)

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

    def train(self):
        for ep in range(self.start_epoch, self.epochs):
            self.train_one_epoch(ep)

            train_acc = self.evaluate_acc(self.train_loader)
            valid_acc = self.evaluate_acc(self.valid_loader)
            print(f'train accuracy: {train_acc}')
            print(f'valid accuracy: {valid_acc}')
            self.writer.add_scalar('Acc/train', train_acc, ep)
            self.writer.add_scalar('Acc/valid', valid_acc, ep)

            if valid_acc > self.best_acc:
                self.best_acc = valid_acc
                torch.save(self.model.state_dict(), os.path.join(self.log_root, 'best_model.pt'))

            if self.save_per_epochs and (ep + 1) % self.save_per_epochs == 0:
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
    def evaluate_acc(self, dataloader):
        self.model.eval()
        evaluator = SimpleClassificationEvaluator()
        for X, y in tqdm(dataloader, desc='Evaluating', ncols=120, leave=False, colour='yellow'):
            X = X.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.long)
            scores = self.model(X)
            _, preds = scores.max(dim=1)
            evaluator.update(preds, y)
        return evaluator.Accuracy()
