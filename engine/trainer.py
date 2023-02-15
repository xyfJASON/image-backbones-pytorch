import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import MulticlassAccuracy

from metrics import AverageMeter
from utils.optimizer import optimizer_to_device
from utils.logger import get_logger, StatusTracker
from utils.data import get_dataloader, get_data_generator
from utils.misc import get_time_str, init_seeds, create_exp_dir, check_freq, get_bare_model
from utils.dist import init_distributed_mode, broadcast_objects, main_process_only
from utils.dist import get_rank, get_world_size, get_local_rank, is_dist_avail_and_initialized
from engine.tools import build_model, build_optimizer, build_scheduler, build_dataset, find_resume_checkpoint


class Trainer:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.time_str = get_time_str()

        # INITIALIZE DISTRIBUTED MODE
        init_distributed_mode()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # INITIALIZE SEEDS
        init_seeds(self.cfg.SEED + get_rank())

        # CREATE EXPERIMENT DIRECTORY
        self.exp_dir = create_exp_dir(
            cfg_dump=self.cfg.dump(),
            resume=self.cfg.TRAIN.RESUME is not None,
            time_str=self.time_str,
            name=self.args.name,
            no_interaction=self.args.no_interaction,
        )
        self.exp_dir = broadcast_objects(self.exp_dir)

        # INITIALIZE LOGGER
        self.logger = get_logger(log_file=os.path.join(self.exp_dir, f'output-{self.time_str}.log'))
        self.logger.info(f'Experiment directory: {self.exp_dir}')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f"Number of devices: {get_world_size()}")

        # BUILD DATASET & DATALOADER & DATA GENERATOR
        train_set = build_dataset(self.cfg, split='train')
        valid_set = build_dataset(self.cfg, split='valid')
        self.train_loader = get_dataloader(
            dataset=train_set,
            shuffle=True,
            drop_last=True,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            prefetch_factor=self.cfg.DATALOADER.PREFETCH_FACTOR,
        )
        self.valid_loader = get_dataloader(
            dataset=valid_set,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            prefetch_factor=self.cfg.DATALOADER.PREFETCH_FACTOR,
        )
        self.micro_batch = self.cfg.DATALOADER.MICRO_BATCH
        if self.micro_batch == 0:
            self.micro_batch = self.cfg.DATALOADER.BATCH_SIZE
        effective_batch = self.cfg.DATALOADER.BATCH_SIZE * get_world_size()
        self.logger.info(f'Size of training set: {len(train_set)}')
        self.logger.info(f'Size of validation set: {len(valid_set)}')
        self.logger.info(f'Batch size per device: {self.cfg.DATALOADER.BATCH_SIZE}')
        self.logger.info(f'Effective batch size: {effective_batch}')

        # BUILD MODEL, OPTIMIZER AND SCHEDULER
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        params = filter(lambda x: x.requires_grad, self.model.parameters())
        self.optimizer = build_optimizer(params, self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

        # LOAD PRETRAINED WEIGHTS
        if self.cfg.MODEL.WEIGHTS is not None:
            weights = torch.load(self.cfg.MODEL.WEIGHTS, map_location='cpu')
            self.model.load_state_dict(weights['model'])
            self.logger.info(f'Successfully load model from {self.cfg.MODEL.WEIGHTS}')

        # RESUME
        self.cur_step = 0
        self.best_acc = 0
        if self.cfg.TRAIN.RESUME is not None:
            resume_path = find_resume_checkpoint(self.exp_dir, self.cfg.TRAIN.RESUME)
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

        # DEFINE LOSSES
        self.cross_entropy = nn.CrossEntropyLoss()
        self.loss_meter = AverageMeter().to(self.device)

        # DEFINE EVALUATION METRICS
        self.metric_acc1 = MulticlassAccuracy(self.cfg.DATA.N_CLASSES, top_k=1, average='micro').to(self.device)
        self.metric_acc5 = MulticlassAccuracy(self.cfg.DATA.N_CLASSES, top_k=5, average='micro').to(self.device)
        self.metric_loss = AverageMeter().to(self.device)

        # DEFINE STATUS TRACKER
        self.status_tracker = StatusTracker(
            logger=self.logger,
            exp_dir=self.exp_dir,
            print_freq=cfg.TRAIN.PRINT_FREQ,
        )

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
        train_data_generator = get_data_generator(
            dataloader=self.train_loader,
            start_epoch=self.cur_step,
        )
        while self.cur_step < self.cfg.TRAIN.TRAIN_STEPS:
            # get a batch of data
            batch = next(train_data_generator)
            # run a step
            train_status = self.run_step(batch)
            self.status_tracker.track_status('Train', train_status, self.cur_step)
            # evaluate
            if check_freq(self.cfg.TRAIN.EVAL_FREQ, self.cur_step):
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
            if check_freq(self.cfg.TRAIN.SAVE_FREQ, self.cur_step):
                self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step:0>6d}.pt'))
            # synchronizes all processes
            if is_dist_avail_and_initialized():
                dist.barrier()
            self.cur_step += 1
        # save the last checkpoint if not saved
        if not check_freq(self.cfg.TRAIN.SAVE_FREQ, self.cur_step - 1):
            self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step-1:0>6d}.pt'))
        self.status_tracker.close()
        self.logger.info(f'Best valid accuracy: {self.best_acc}')
        self.logger.info('End of training')

    def run_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        self.loss_meter.reset()
        batchX, batchy = batch
        batch_size = batchX.shape[0]
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
            self.loss_meter.update(loss.detach(), X.shape[0])
        train_status = dict(
            loss=self.loss_meter.compute(),
            lr=self.optimizer.param_groups[0]['lr'],
        )
        self.optimizer.step()
        self.scheduler.step()
        return train_status

    @torch.no_grad()
    def evaluate(self, dataloader):
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
