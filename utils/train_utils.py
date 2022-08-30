import os
import sys
import shutil
import random
import datetime
import numpy as np

import torch
from torch.backends import cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.optimizer import optimizer_to_device


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def set_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:                   # multiple gpus
            dist.init_process_group(backend='nccl')
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            global_rank = int(os.environ['RANK'])
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:                                               # single gpu
            world_size, local_rank, global_rank = 1, 0, 0
            device = torch.device('cuda')
    else:                                                   # cpu
        world_size, local_rank, global_rank = 0, -1, -1
        device = torch.device('cpu')
    return device, world_size, local_rank, global_rank


def create_log_directory(config, config_path):
    if config.get('resume', None) and config['resume']['log_root'] is not None:
        assert os.path.isdir(config['resume']['log_root'])
        log_root = config['resume']['log_root']
    else:
        log_root = os.path.join('runs', datetime.datetime.now().strftime('exp-%Y-%m-%d-%H-%M-%S'))
        os.makedirs(log_root)
    print('log directory:', log_root)

    if config.get('save_freq'):
        os.makedirs(os.path.join(log_root, 'ckpt'), exist_ok=True)
    if config.get('sample_freq'):
        os.makedirs(os.path.join(log_root, 'samples'), exist_ok=True)
    if os.path.exists(os.path.join(log_root, 'config.yml')):
        print('Warning: config.yml exists and will be replaced by a new one.', file=sys.stderr)
    shutil.copyfile(config_path, os.path.join(log_root, 'config.yml'))
    return log_root


def load_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device=device)
    return model


def save_model(save_path, model):
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)


def load_ckpt(resume_path, model, optimizer, scheduler, device):
    ckpt = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device=device)
    optimizer.load_state_dict(ckpt['optimizer'])
    optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    print(f'Load checkpoint at epoch {start_epoch - 1}.')
    print(f'Best accuracy so far: {best_acc}')
    return start_epoch, best_acc


def save_ckpt(save_path, model, optimizer, scheduler, epoch, best_acc):
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc},
               save_path)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
