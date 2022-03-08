import os
import shutil
import yaml
import torch


def optimizer_load_state_dict(optimizer, ckpt, device):
    optimizer.load_state_dict(ckpt)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def parse_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config['exp_name'] is None:
        raise ValueError('exp_name missing')

    device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    logroot = os.path.join('runs', config['exp_name'])
    print('logroot:', logroot)
    if os.path.exists(logroot):
        if (config.get('resume_path') is None) or (os.path.realpath(config['resume_path']).find(os.path.realpath(logroot)) == -1):
            if input('logroot path already exists. Cover it anyway? y/n: ') == 'y':
                shutil.rmtree(logroot)
            else:
                exit()

    if config.get('save_per_epochs') and not os.path.exists(os.path.join(logroot, 'ckpt')):
        os.makedirs(os.path.join(logroot, 'ckpt'))

    if not os.path.exists(os.path.join(logroot, 'tensorboard')):
        os.makedirs(os.path.join(logroot, 'tensorboard'))

    shutil.copyfile('./config.yml', os.path.join(logroot, 'config.yml'))

    return config, device, logroot
