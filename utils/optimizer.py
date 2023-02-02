from yacs.config import CfgNode as CN

import torch
import torch.optim as optim


def optimizer_to_device(optimizer: optim.Optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def build_optimizer(params, cfg: CN) -> optim.Optimizer:
    """
    Args:
        params: parameter groups of model
        cfg (CN): an instance of CfgNode with attributes:
            - TRAIN.OPTIM.NAME: 'SGD', 'RMSProp', 'Adam', 'AdamW', etc.
            - TRAIN.OPTIM.LR
            - TRAIN.OPTIM.WEIGHT_DECAY
            - ... (other arguments of the optimizer)
    """
    cfg = cfg.TRAIN.OPTIM
    if cfg.NAME == 'SGD':
        optimizer = optim.SGD(
            params=params,
            lr=cfg.LR,
            momentum=getattr(cfg, 'MOMENTUM', 0),
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0),
            nesterov=getattr(cfg, 'NESTEROV', False),
        )
    elif cfg.NAME == 'Adam':
        optimizer = optim.Adam(
            params=params,
            lr=cfg.LR,
            betas=getattr(cfg, 'BETAS', (0.9, 0.999)),
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0),
        )
    else:
        raise ValueError(f"Optimizer {cfg.NAME} is not supported.")
    return optimizer


def _test():
    model = torch.nn.Linear(20, 10)

    cfg = CN()
    cfg.TRAIN = CN()
    cfg.TRAIN.OPTIM = CN()
    cfg.TRAIN.OPTIM.NAME = 'SGD'
    cfg.TRAIN.OPTIM.LR = 0.001
    cfg.TRAIN.OPTIM.MOMENTUM = 0.9
    cfg.TRAIN.OPTIM.WEIGHT_DECAY = 0.05
    cfg.TRAIN.OPTIM.NESTEROV = True
    optimizer = build_optimizer(model.parameters(), cfg)
    print(optimizer)

    cfg.TRAIN.OPTIM.NAME = 'Adam'
    optimizer = build_optimizer(model.parameters(), cfg)
    print(optimizer)


if __name__ == '__main__':
    _test()
