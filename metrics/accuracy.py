from typing import Sequence

from torch import Tensor


def reduce_tensor(x: Tensor, reduction: str):
    assert reduction in ['mean', 'sum', 'none'], f'Reduction {reduction} not implemented'
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()
    return x


def accuracy(output: Tensor, target: Tensor, topk: Sequence[int] = (1, )):
    """Computes the accuracy over the k top predictions

    Args:
        output: model output (logits), Tensor of shape (batch_size, num_classes)
        target: true labels, Tensor of shape (batch_size, )
        topk: list of integers, topk values to calculate accuracy

    Returns:
        For each k, return a bool Tensor of shape (batch_size, ) where 1 indicates correct prediction

    References:
      https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py
    """
    maxk = min(max(topk), output.shape[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.T
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].sum(dim=0).bool() for k in topk]


class Accuracy:
    def __init__(self, topk: Sequence[int] = (1, ), reduction: str = 'mean'):
        self.topk = topk
        self.reduction = reduction

    def __call__(self, output: Tensor, target: Tensor):
        correct = accuracy(output, target, self.topk)
        return [reduce_tensor(c.float(), self.reduction) for c in correct]
