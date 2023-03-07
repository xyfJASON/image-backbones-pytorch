# import torch
# from torchmetrics import Metric


# class AverageMeter(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#
#     def update(self, val: torch.Tensor, n: int):
#         self.sum += val * n
#         self.total += n
#
#     def compute(self):
#         return self.sum / self.total

class AverageMeter:
    """
    Computes and stores the average and current value

    Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
