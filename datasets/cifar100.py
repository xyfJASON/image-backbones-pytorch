import torchvision.transforms as T
import torchvision.datasets as dset


CIFAR100 = dset.CIFAR100


def get_default_transforms(img_size: int, split: str):
    """
    Default transforms are:
     - RandomCrop
     - RandomHorizontalFlip
     - Normalize to 0-mean and 1-std
    """
    assert img_size == 32, f'Default transform for cifar100 only supports 32x32 image size, get {img_size}.'
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    if split == 'train':
        transforms = T.Compose([T.RandomCrop(32, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize(mean, std)])
    else:
        transforms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transforms
