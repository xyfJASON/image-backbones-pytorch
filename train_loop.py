import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MulticlassAccuracy

from metrics import AverageMeter
from utils.logger import StatusTracker
from utils.data import get_data_generator
from utils.optimizer import optimizer_to_device
from utils.misc import check_freq, get_bare_model
from utils.dist import is_dist_avail_and_initialized, main_process_only, get_local_rank, all_reduce_mean


class TrainLoop:
    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            device,
            exp_dir,
            logger,
            batch_size,
            micro_batch,
            train_steps,
            resume,
            print_freq,
            save_freq,
            eval_freq,
            use_fp16,
            **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.exp_dir = exp_dir
        self.logger = logger
        self.batch_size = batch_size
        self.micro_batch = micro_batch
        self.train_steps = train_steps
        self.resume = resume
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.use_fp16 = use_fp16

        assert use_fp16 is False, 'use_fp16 is not supported for now'
        if micro_batch == 0:
            self.micro_batch = self.batch_size

        # RESUME
        self.cur_step = 0
        self.best_acc = 0
        if self.resume is not None:
            resume_path = find_resume_checkpoint(self.exp_dir, self.resume)
            self.logger.info(f'Resume from {resume_path}')
            self.load_ckpt(resume_path)

        # DISTRIBUTED MODELS
        if is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(
                module=self.model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                broadcast_buffers=True,
                find_unused_parameters=False,
            )

        # GET DATA GENERATOR
        # XXX: `start_epoch` is used to set random seed for dataloader sampler in distributed mode,
        # XXX: thus passing `cur_step` also does the work. However, it leads to less reproducibility
        # XXX: if two runs resume differently, even with the same seed.
        self.train_data_generator = get_data_generator(self.train_loader, start_epoch=self.cur_step)

        # DEFINE LOSSES
        self.cross_entropy = nn.CrossEntropyLoss()

        # DEFINE EVALUATION METRICS
        self.metric_acc1 = MulticlassAccuracy(kwargs['n_classes'], top_k=1, average='micro').to(self.device)
        self.metric_acc5 = MulticlassAccuracy(kwargs['n_classes'], top_k=5, average='micro').to(self.device)
        self.metric_loss = AverageMeter().to(self.device)

        # DEFINE STATUS TRACKER
        self.status_tracker = StatusTracker(logger=self.logger, exp_dir=self.exp_dir, print_freq=self.print_freq)

    def load_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # load model
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.logger.info(f'Successfully load model from {ckpt_path}')
        # load optimizer
        self.optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_to_device(self.optimizer, self.device)
        self.logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load scheduler
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.logger.info(f'Successfully load scheduler from {ckpt_path}')
        # load meta informations
        self.cur_step = ckpt['step'] + 1
        self.best_acc = ckpt['best_acc']
        self.logger.info(f'Restart training at step {self.cur_step}')
        self.logger.info(f'Best accuracy so far: {self.best_acc}')

    @main_process_only
    def save_ckpt(self, save_path: str):
        state_dicts = dict(
            model=get_bare_model(self.model).state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            step=self.cur_step,
            best_acc=self.best_acc,
        )
        torch.save(state_dicts, save_path)

    def run_loop(self):
        self.logger.info('Start training...')
        while self.cur_step < self.train_steps:
            # get a batch of data
            batch = next(self.train_data_generator)
            # run a step
            self.run_step(batch)
            # evaluate
            if check_freq(self.eval_freq, self.cur_step):
                # evaluate on training set
                eval_status_train = self.evaluate(self.train_loader)
                eval_status_train.pop('loss')
                eval_status_train = {f'{k}(train_set)': v for k, v in eval_status_train.items()}
                self.status_tracker.track_status('Eval', eval_status_train, self.cur_step)
                # evaluate on validation set
                eval_status_valid = self.evaluate(self.valid_loader)
                eval_status_valid = {f'{k}(valid_set)': v for k, v in eval_status_valid.items()}
                self.status_tracker.track_status('Eval', eval_status_valid, self.cur_step)
                # save the best model
                if eval_status_valid['acc@1(valid_set)'] > self.best_acc:
                    self.best_acc = eval_status_valid['acc@1(valid_set)']
                    self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', 'best.pt'))
            # save checkpoint
            if check_freq(self.save_freq, self.cur_step):
                self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step:0>6d}.pt'))
            # synchronizes all processes
            if is_dist_avail_and_initialized():
                dist.barrier()
            self.cur_step += 1
        # save the last checkpoint if not saved
        if not check_freq(self.save_freq, self.cur_step - 1):
            self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step-1:0>6d}.pt'))
        self.logger.info(f'Best valid accuracy: {self.best_acc}')
        self.logger.info('End of training')

    def run_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        batchX, batchy = batch
        batch_size = batchX.shape[0]
        loss_value = 0.
        for i in range(0, batch_size, self.micro_batch):
            X = batchX[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            y = batchy[i:i+self.micro_batch].to(device=self.device, dtype=torch.long)
            # no need to synchronize gradient before the last micro batch
            no_sync = is_dist_avail_and_initialized() and (i + self.micro_batch) < batch_size
            cm = self.model.no_sync() if no_sync else nullcontext()
            with cm:
                scores = self.model(X)
                loss = self.cross_entropy(scores, y)
                loss.backward()
            loss_value += loss.detach() * X.shape[0]
        loss_value /= batch_size
        loss_value = all_reduce_mean(loss_value)
        train_status = dict(
            loss=loss_value.item(),
            lr=self.optimizer.param_groups[0]['lr'],
        )
        self.status_tracker.track_status('Train', train_status, self.cur_step)
        self.optimizer.step()
        self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.valid_loader
        self.model.eval()
        self.metric_acc1.reset()
        self.metric_acc5.reset()
        self.metric_loss.reset()
        for batchX, batchy in dataloader:
            batch_size = batchX.shape[0]
            for i in range(0, batch_size, self.micro_batch):
                X = batchX[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
                y = batchy[i:i+self.micro_batch].to(device=self.device, dtype=torch.long)
                scores = self.model(X)
                self.metric_acc1.update(scores, y)
                self.metric_acc5.update(scores, y)
                loss = self.cross_entropy(scores, y)
                self.metric_loss.update(loss, X.shape[0])
        return {
            'acc@1': self.metric_acc1.compute().item(),
            'acc@5': self.metric_acc5.compute().item(),
            'loss': self.metric_loss.compute().item(),
        }


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
