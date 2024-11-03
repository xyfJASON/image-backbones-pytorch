import os
import sys
import tqdm
import shutil
import datetime
from typing import List


def check_freq(freq: int, step: int):
    assert isinstance(freq, int) and freq >= 0
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def get_dataloader_iterator(dataloader, tqdm_kwargs):
    while True:
        for batch in tqdm.tqdm(dataloader, **tqdm_kwargs):
            yield batch


def find_resume_checkpoint(exp_dir: str, resume: str):
    """ Checkpoints are named after 'stepxxxxxx/' """
    if os.path.isdir(resume):
        ckpt_path = resume
    elif resume == 'best':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'best')
    elif resume == 'latest':
        d = dict()
        for name in os.listdir(os.path.join(exp_dir, 'ckpt')):
            if os.path.isdir(os.path.join(exp_dir, 'ckpt', name)) and name[:4] == 'step':
                d.update({int(name[4:]): name})
        ckpt_path = os.path.join(exp_dir, 'ckpt', d[sorted(d)[-1]])
    else:
        raise ValueError(f'resume option {resume} is invalid')
    assert os.path.isdir(ckpt_path), f'{ckpt_path} is not a directory'
    return ckpt_path


def create_exp_dir(
        exp_dir: str,
        conf_yaml: str,
        subdirs: List[str] = ('ckpt', ),
        time_str: str = None,
        exist_ok: bool = False,
        cover_dir: bool = False,
):
    """Create the experiment directory.

    Args:
        exp_dir: The path to the experiment directory.
        conf_yaml: A string of the configuration in YAML format.
        subdirs: The subdirectories to create in the experiment directory.
        time_str: The time string to append to the configuration file name.
        exist_ok: Whether to allow the directory to exist. Note that some files may be overwritten if True.
        cover_dir: Whether to cover the directory if it already exists. Note that all files will be removed if True.
    """
    # check if the directory exists
    if os.path.exists(exp_dir) and not exist_ok:
        cover = cover_dir or query_yes_no(
            question=f'{exp_dir} already exists! Cover it anyway?',
            default='no',
        )
        shutil.rmtree(exp_dir, ignore_errors=True) if cover else sys.exit(1)

    # make directories
    os.makedirs(exp_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    # write configuration
    if time_str is None:
        time_str = get_time_str()
    with open(os.path.join(exp_dir, f'config-{time_str}.yaml'), 'w') as f:
        f.write(conf_yaml)


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
