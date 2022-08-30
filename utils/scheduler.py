import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def build_scheduler(optimizer, cfg):
    """
    Args:
        optimizer: the optimizer for which to schedule the learning rate
        cfg: configuration dictionary, whose keys include:
            - choice: 'cosineannealinglr' or 'multisteplr'
            - cosineannealinglr:
                - T_max
                - eta_min
            - multisteplr:
                - milestones
                - gamma
            - warmup:
                - warmup_epochs
                - warmup_factor
    """
    if cfg['choice'] == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=cfg['cosineannealinglr']['T_max'],
                                      eta_min=cfg['cosineannealinglr']['eta_min'])

    elif cfg['choice'] == 'multisteplr':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg['multisteplr']['milestones'],
                                gamma=cfg['multisteplr']['gamma'])

    else:
        raise ValueError(f"Scheduler {cfg['choice']} is not supported.")

    if cfg.get('warmup', None) and cfg['warmup'].get('warmup_epochs', 0) > 0:
        scheduler = LRWarmupWrapper(scheduler,
                                    warmup_epochs=cfg['warmup']['warmup_epochs'],
                                    warmup_factor=cfg['warmup']['warmup_factor'])

    return scheduler


class LRWarmupWrapper:
    """
    This class wraps the standard PyTorch LR scheduler to support warmup.
    Simplified based on https://github.com/serend1p1ty/core-pytorch-utils/blob/main/cpu/lr_scheduler.py
    """
    def __init__(self, torch_scheduler, warmup_epochs: int = 0, warmup_factor: float = 0.1):
        """
        Args:
            torch_scheduler: torch.optim.lr_scheduler._LRScheduler
            warmup_epochs (int): How many epochs in warmup stage. Defaults to 0 to disable warmup.
            warmup_factor (float): The factor of initial warmup lr relative to base lr. Defaults to 0.1.
        """
        self.torch_scheduler = torch_scheduler
        assert isinstance(warmup_epochs, int) and isinstance(warmup_factor, float)
        assert warmup_epochs > 0 and warmup_factor < 1, f'warmup only works when warmup_epochs > 0 and warmup_factor < 1'
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor

        self.param_groups = self.torch_scheduler.optimizer.param_groups
        self.base_lrs = [param_group['lr'] for param_group in self.param_groups]
        self.regular_lrs_in_warmup = self._pre_compute_regular_lrs_in_warmup()

        self.last_epoch = 0

        self._set_lrs([base_lr * warmup_factor for base_lr in self.base_lrs])

    def _pre_compute_regular_lrs_in_warmup(self):
        regular_lrs_in_warmup = [self.base_lrs]
        for _ in range(self.warmup_epochs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.torch_scheduler.step()
            regular_lrs_in_warmup.append([param_group['lr'] for param_group in self.param_groups])
        return regular_lrs_in_warmup

    def _set_lrs(self, lrs):
        for param_group, lr in zip(self.param_groups, lrs):
            param_group['lr'] = lr

    def _get_warmup_lrs(self, epoch, regular_lrs):
        alpha = epoch / self.warmup_epochs
        factor = self.warmup_factor * (1 - alpha) + alpha
        return [lr * factor for lr in regular_lrs]

    def step(self):
        self.last_epoch += 1
        if self.last_epoch < self.warmup_epochs:
            self._set_lrs(self._get_warmup_lrs(self.last_epoch, self.regular_lrs_in_warmup[self.last_epoch]))
        elif self.last_epoch == self.warmup_epochs:
            self._set_lrs(self.regular_lrs_in_warmup[-1])
        else:
            self.torch_scheduler.step()

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key != "torch_scheduler"}
        state["torch_scheduler"] = self.torch_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.torch_scheduler.load_state_dict(state_dict.pop("torch_scheduler"))
        self.__dict__.update(state_dict)


def _test(choice='cosineannealinglr'):
    model = nn.Linear(3, 4)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer_warmup = optim.SGD(model.parameters(), lr=0.1)

    cfg = dict(choice=choice,
               cosineannealinglr=dict(T_max=100, eta_min=0.001),
               multisteplr=dict(milestones=[60, 80], gamma=0.1))
    scheduler = build_scheduler(optimizer, cfg)
    scheduler_warmup = build_scheduler(optimizer_warmup, cfg)
    scheduler_warmup = LRWarmupWrapper(scheduler_warmup, warmup_epochs=40, warmup_factor=0.01)

    epochs = range(100)
    lrs, lrs_warmup = [], []
    for _ in epochs:
        lrs.append(optimizer.param_groups[0]['lr'])
        lrs_warmup.append(optimizer_warmup.param_groups[0]['lr'])
        scheduler.step()
        scheduler_warmup.step()
    print(lrs, lrs_warmup)
    plt.plot(epochs, lrs)
    plt.plot(epochs, lrs_warmup)
    plt.show()


if __name__ == '__main__':
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    _test('cosineannealinglr')
    _test('multisteplr')
