from omegaconf import DictConfig

import torchvision.datasets as dset
import torchvision.transforms as T


def load_data(conf: DictConfig, split='train'):
    """Keys in conf: 'name', 'dataroot', 'img_size'."""
    assert conf.get('name') is not None
    assert conf.get('dataroot') is not None
    assert conf.get('img_size') is not None

    if conf.name.lower() in ['cifar10', 'cifar-10']:
        assert conf.img_size == 32
        crop = T.RandomCrop(32, padding=4) if split == 'train' else T.CenterCrop(32)
        flip_p = 0.5 if split == 'train' else 0.0
        transform = T.Compose([
            crop,
            T.RandomHorizontalFlip(p=flip_p),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
        dataset = dset.CIFAR10(root=conf.dataroot, train=(split == 'train'), transform=transform)

    elif conf.name.lower() in ['cifar100', 'cifar-100']:
        assert conf.img_size == 32
        crop = T.RandomCrop(32, padding=4) if split == 'train' else T.CenterCrop(32)
        flip_p = 0.5 if split == 'train' else 0.0
        transform = T.Compose([
            crop,
            T.RandomHorizontalFlip(p=flip_p),
            T.ToTensor(),
            T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
        dataset = dset.CIFAR100(root=conf.dataroot, train=(split == 'train'), transform=transform)

    else:
        raise ValueError(f'Unsupported dataset: {conf.name}')

    return dataset
