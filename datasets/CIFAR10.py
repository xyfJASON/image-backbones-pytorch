import torchvision.transforms as T
import torchvision.datasets as dset


CIFAR10 = dset.CIFAR10


def get_default_transforms(img_size: int, split: str):
    """
    Default transforms are:
     - RandomCrop
     - RandomHorizontalFlip
     - Normalize to 0-mean and 1-std
    """
    assert img_size == 32, f'Default transform for cifar10 only supports 32x32 image size, get {img_size}.'
    mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
    if split == 'train':
        transforms = T.Compose([T.RandomCrop(32, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean, std)])
    else:
        transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transforms
