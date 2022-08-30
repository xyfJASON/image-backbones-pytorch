from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torchvision.datasets as dset


def build_dataset(name, dataroot, img_size, split, transforms=None):
    """
    Supported datasets:
        - cifar10
        - cifar100

    Default transforms are RandomCrop / Resize with RandomHorizontalFlip,
    and images will be normalized to be 0-mean and 1-std
    """
    if name == 'cifar10':
        assert split in ['train', 'test'], f'cifar10 only has train/test split, get {split}.'
        if transforms is None:
            assert img_size == 32, f'Default transform for cifar10 only supports 32x32 image size, get {img_size}.'
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            if split == 'train':
                transforms = T.Compose([T.RandomCrop(32, padding=4),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        T.Normalize(mean, std)])
            else:
                transforms = T.Compose([T.ToTensor(),
                                        T.Normalize(mean, std)])
        dataset = dset.CIFAR10(root=dataroot, train=(split == 'train'), transform=transforms)
        n_classes = 10

    elif name == 'cifar100':
        assert split in ['train', 'test'], f'cifar100 only has train/test split, get {split}.'
        if transforms is None:
            assert img_size == 32, f'Default transform for cifar100 only supports 32x32 image size, get {img_size}.'
            mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            if split == 'train':
                transforms = T.Compose([T.RandomCrop(32, padding=4),
                                        T.RandomHorizontalFlip(),
                                        T.ToTensor(),
                                        T.Normalize(mean, std)])
            else:
                transforms = T.Compose([T.ToTensor(),
                                        T.Normalize(mean, std)])
        dataset = dset.CIFAR100(root=dataroot, train=(split == 'train'), transform=transforms)
        n_classes = 100

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    return dataset, n_classes


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False, is_ddp=False):
    if is_ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader
