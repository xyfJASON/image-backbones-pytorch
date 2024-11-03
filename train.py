import os
import tqdm
import argparse
from omegaconf import OmegaConf
from contextlib import nullcontext

import accelerate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import Accuracy
from tools import build_model, build_optimizer, build_scheduler
from utils.data import get_dataset
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint, get_dataloader_iterator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
    parser.add_argument('-cd', '--cover_dir', action='store_true', default=False, help='Cover the experiment directory if it exists')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator(step_scheduler_with_optimizer=False)
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), subdirs=['ckpt'],
            time_str=args.time_str, exist_ok=args.resume is not None, cover_dir=args.cover_dir,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=accelerator.is_main_process,
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, 'tensorboard'),
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert conf.train.batch_size % accelerator.num_processes == 0
    bspp = conf.train.batch_size // accelerator.num_processes
    micro_batch_size = conf.train.micro_batch_size or bspp
    train_set = get_dataset(name=conf.data.name, dataroot=conf.data.dataroot, img_size=conf.data.img_size, split='train')
    valid_set = get_dataset(name=conf.data.name, dataroot=conf.data.dataroot, img_size=conf.data.img_size, split='valid')
    train_loader = DataLoader(train_set, batch_size=bspp, shuffle=True, drop_last=True, **conf.dataloader)
    valid_loader = DataLoader(valid_set, batch_size=micro_batch_size, shuffle=False, drop_last=False, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Size of validation set: {len(valid_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL, OPTIMIZER AND SCHEDULER
    model = build_model(conf)
    optimizer = build_optimizer(model.parameters(), conf)
    scheduler = build_scheduler(optimizer, conf)
    logger.info(f'Number of parameters of transformer: {sum(p.numel() for p in model.parameters()):,}')
    logger.info('=' * 50)

    # RESUME TRAINING
    step, best_acc = 0, 0.
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load model
        ckpt_model = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {resume_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(resume_path, 'optimizer.pt'), map_location='cpu', weights_only=True)
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {resume_path}')
        # load scheduler
        ckpt_scheduler = torch.load(os.path.join(resume_path, 'scheduler.pt'), map_location='cpu', weights_only=True)
        scheduler.load_state_dict(ckpt_scheduler['scheduler'])
        logger.info(f'Successfully load scheduler from {resume_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(resume_path, 'meta.pt'), map_location='cpu', weights_only=True)
        step = ckpt_meta['step'] + 1
        best_acc = ckpt_meta['best_acc']
        logger.info(f'Restart training at step {step}')
        logger.info(f'Best accuracy so far: {best_acc}')
        del ckpt_model, ckpt_optimizer, ckpt_scheduler, ckpt_meta

    # PREPARE FOR DISTRIBUTED TRAINING AND MIXED PRECISION
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, optimizer, scheduler, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader,  # type: ignore
    )
    unwrapped_model = accelerator.unwrap_model(model)

    # DEFINE LOSSES AND METRICS
    cross_entropy = nn.CrossEntropyLoss()
    accuracy_fn = Accuracy(topk=(1, 5), reduction='none')
    accelerator.wait_for_everyone()

    @accelerator.on_local_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model
        model_state_dicts = dict(model=unwrapped_model.state_dict())
        accelerator.save(model_state_dicts, os.path.join(save_path, 'model.pt'))
        # save optimizer
        optimizer_state_dicts = dict(optimizer=optimizer.state_dict())
        accelerator.save(optimizer_state_dicts, os.path.join(save_path, 'optimizer.pt'))
        # save scheduler
        scheduler_state_dicts = dict(scheduler=scheduler.state_dict())
        accelerator.save(scheduler_state_dicts, os.path.join(save_path, 'scheduler.pt'))
        # save meta information
        meta_state_dicts = dict(step=step, best_acc=best_acc)
        accelerator.save(meta_state_dicts, os.path.join(save_path, 'meta.pt'))

    def run_step(batch):
        # get data
        batchx, batchy = batch
        batch_size = batchx.shape[0]
        # gradient accumulation
        for i in range(0, batch_size, micro_batch_size):
            x = batchx[i:i+micro_batch_size].float()
            y = batchy[i:i+micro_batch_size].long()
            loss_scale = x.shape[0] / batch_size
            no_sync = (i + micro_batch_size) < batch_size
            with accelerator.no_sync(model) if no_sync else nullcontext():
                scores = model(x)
                loss = cross_entropy(scores, y)
                accelerator.backward(loss * loss_scale)
        # optimize
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return dict(
            loss=loss.item(),
            lr=optimizer.param_groups[0]['lr'],
        )

    @torch.no_grad()
    def evaluate(dataloader):
        acc1_list, acc5_list = [], []
        for x, y in tqdm.tqdm(
            dataloader, desc='Evaluating', leave=False,
            disable=not accelerator.is_main_process,
        ):
            x, y = x.float(), y.long()
            scores = model(x)
            acc1, acc5 = accuracy_fn(scores, y)
            acc1 = accelerator.gather_for_metrics(acc1)
            acc5 = accelerator.gather_for_metrics(acc5)
            acc1_list.append(acc1)
            acc5_list.append(acc5)
        acc1_list = torch.cat(acc1_list).cpu()
        acc5_list = torch.cat(acc5_list).cpu()
        return {
            'acc@1': acc1_list.mean().item(),
            'acc@5': acc5_list.mean().item(),
        }

    # START TRAINING
    logger.info('Start training...')
    train_loader_iterator = get_dataloader_iterator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        _batch = next(train_loader_iterator)
        # run a step
        model.train()
        train_status = run_step(_batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()
        # validate
        model.eval()
        # evaluate
        if check_freq(conf.train.eval_freq, step):
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
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    logger.info(f'Best valid accuracy: {best_acc:.4f}')
    accelerator.wait_for_everyone()
    status_tracker.close()
    accelerator.end_training()
    logger.info('End of training')


if __name__ == '__main__':
    main()
