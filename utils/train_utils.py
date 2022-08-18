import os
import random
import numpy as np

import torch
from torch.backends import cudnn
import torch.distributed as dist


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


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
