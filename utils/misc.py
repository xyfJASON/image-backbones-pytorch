import os
import sys
import random
import shutil
import datetime
import numpy as np
from yacs.config import CfgNode as CN

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.dist import main_process_only
from configs.defaults import get_cfg_defaults


def init_seeds(seed: int = 0, cuda_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def check_freq(freq: int, step: int):
    assert isinstance(freq, int)
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def get_bare_model(model: nn.Module or DDP):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


@main_process_only
def create_exp_dir(cfg: CN, time_str: str = None, name: str = None, no_interaction: bool = False):
    if time_str is None:
        time_str = get_time_str()
    if name is None:
        name = f'exp-{time_str}'
    exp_dir = os.path.join('runs', name)

    if os.path.exists(exp_dir) and getattr(cfg.TRAIN, 'RESUME', None) is None:
        cover = True if no_interaction else query_yes_no(f'{exp_dir} already exists! Cover it anyway?', default='no')
        if cover:
            shutil.rmtree(exp_dir, ignore_errors=True)
        else:
            sys.exit(1)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'ckpt'), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)

    with open(os.path.join(exp_dir, f'config-{time_str}.yaml'), 'w') as f:
        f.write(cfg.dump())
    return exp_dir


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Copied from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
