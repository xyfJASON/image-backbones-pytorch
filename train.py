import os
import tqdm
import argparse
from contextlib import nullcontext
from yacs.config import CfgNode as CN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import accelerate

from metrics import AverageMeter, accuracy
from utils.logger import StatusTracker, get_logger
from utils.data import get_dataset, get_data_generator
from tools import build_model, build_optimizer, build_scheduler
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to config file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/{current time}/',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def train(args, cfg):
    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    print(f'Process {accelerator.process_index} '
          f'using device: {accelerator.device}')
    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_local_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            cfg_dump=cfg.dump(),
            exist_ok=cfg.train.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_local_main_process,
    )
    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=cfg.train.print_freq,
        is_main_process=accelerator.is_local_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER & DATA GENERATOR
    assert cfg.train.batch_size % accelerator.num_processes == 0
    batch_size_per_process = cfg.train.batch_size // accelerator.num_processes
    micro_batch = cfg.dataloader.micro_batch
    if micro_batch == 0:
        micro_batch = batch_size_per_process
    train_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='train',
    )
    valid_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='valid',
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size_per_process,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=False,
        drop_last=False,
        batch_size=micro_batch,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Size of validation set: {len(valid_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {cfg.train.batch_size}')

    # BUILD MODEL, OPTIMIZER AND SCHEDULER
    model = build_model(cfg)
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)
    step, best_acc = 0, 0.

    def load_ckpt(ckpt_path: str):
        nonlocal step, best_acc
        # load model
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {ckpt_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load scheduler
        ckpt_scheduler = torch.load(os.path.join(ckpt_path, 'scheduler.pt'), map_location='cpu')
        scheduler.load_state_dict(ckpt_scheduler['scheduler'])
        logger.info(f'Successfully load scheduler from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1
        best_acc = ckpt_meta['best_acc']
        logger.info(f'Restart training at step {step}')
        logger.info(f'Best accuracy so far: {best_acc}')

    def save_ckpt(save_path: str):
        if accelerator.is_local_main_process:
            os.makedirs(save_path, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            model_state_dicts = dict(model=unwrapped_model.state_dict())
            accelerator.save(model_state_dicts, os.path.join(save_path, 'model.pt'))
            optimizer_state_dicts = dict(optimizer=optimizer.state_dict())
            accelerator.save(optimizer_state_dicts, os.path.join(save_path, 'optimizer.pt'))
            scheduler_state_dicts = dict(scheduler=scheduler.state_dict())
            accelerator.save(scheduler_state_dicts, os.path.join(save_path, 'scheduler.pt'))
            meta_state_dicts = dict(step=step, best_acc=best_acc)
            accelerator.save(meta_state_dicts, os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
    # PREPARE FOR DISTRIBUTED TRAINING AND MIXED PRECISION
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader,  # type: ignore
    )
    # DEFINE LOSSES
    cross_entropy = nn.CrossEntropyLoss()

    accelerator.wait_for_everyone()

    def run_step(_batch):
        optimizer.zero_grad()
        loss_meter = AverageMeter()
        batchX, batchy = _batch
        batch_size = batchX.shape[0]
        for i in range(0, batch_size, micro_batch):
            X = batchX[i:i+micro_batch].float()
            y = batchy[i:i+micro_batch].long()
            loss_scale = X.shape[0] / batch_size
            no_sync = (i + micro_batch) < batch_size
            cm = accelerator.no_sync(model) if no_sync else nullcontext()
            with cm:
                scores = model(X)
                loss = cross_entropy(scores, y)
                accelerator.backward(loss * loss_scale)
            loss_meter.update(loss.item(), X.shape[0])
        optimizer.step()
        scheduler.step()
        return dict(
            loss=loss_meter.avg,
            lr=optimizer.param_groups[0]['lr'],
        )

    @torch.no_grad()
    def evaluate(dataloader):
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        loss_meter = AverageMeter()
        for X, y in tqdm.tqdm(dataloader, desc='Evaluating', leave=False,
                              disable=not accelerator.is_main_process):
            X, y = X.float(), y.long()
            scores = model(X)
            acc1, acc5 = accuracy(scores, y, topk=(1, 5))
            loss = cross_entropy(scores, y)
            acc1 = accelerator.reduce(acc1, reduction='mean')
            acc5 = accelerator.reduce(acc5, reduction='mean')
            loss = accelerator.reduce(loss, reduction='mean')
            acc1_meter.update(acc1.item(), X.shape[0])
            acc5_meter.update(acc5.item(), X.shape[0])
            loss_meter.update(loss.item(), X.shape[0])
        return {
            'acc@1': acc1_meter.avg,
            'acc@5': acc5_meter.avg,
            'loss': loss_meter.avg,
        }

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
    )
    while step < cfg.train.train_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        model.eval()
        # evaluate
        if check_freq(cfg.train.eval_freq, step):
            # evaluate on training set
            eval_status_train = evaluate(train_loader)
            eval_status_train = {f'{k}(train_set)': v for k, v in eval_status_train.items()}
            status_tracker.track_status('Eval', eval_status_train, step)
            # evaluate on validation set
            eval_status_valid = evaluate(valid_loader)
            eval_status_valid = {f'{k}(valid_set)': v for k, v in eval_status_valid.items()}
            status_tracker.track_status('Eval', eval_status_valid, step)
            # save the best model
            if eval_status_valid['acc@1(valid_set)'] > best_acc:
                best_acc = eval_status_valid['acc@1(valid_set)']
                save_ckpt(os.path.join(exp_dir, 'ckpt', 'best'))
            accelerator.wait_for_everyone()
        # save checkpoint
        if check_freq(cfg.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(cfg.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info(f'Best valid accuracy: {best_acc}')
    logger.info('End of training')


def main():
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    train(args, cfg)


if __name__ == '__main__':
    main()
