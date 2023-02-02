from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from utils.logger import get_logger
from utils.dist import is_dist_avail_and_initialized


available_split = {
    'cifar10': ['train', 'test'],
    'cifar-10': ['train', 'test'],
    'cifar100': ['train', 'test'],
    'cifar-100': ['train', 'test'],
}


def check_split(name, split, strict_valid_test):
    assert split in ['train', 'valid', 'test', 'all']
    if split in ['train', 'all'] or strict_valid_test:
        assert split in available_split[name.lower()], f"Dataset {name} doesn't have split: {split}"
    elif split not in available_split[name.lower()]:
        replace_split = 'test' if split == 'valid' else 'valid'
        assert replace_split in available_split[name.lower()], f"Dataset {name} doesn't have split: {split}"
        logger = get_logger()
        logger.warning(f'Replace split `{split}` with split `{replace_split}`')
        split = replace_split
    return split


def build_dataset(name, dataroot, img_size, split, transforms=None, subset_ids=None, strict_valid_test=False):
    """
    Args:
        name: name of dataset. Options: 'CIFAR-10', 'CIFAR-100'
        dataroot: path to data
        img_size: target size of image
        split: 'train', 'valid', 'test', 'all'
        transforms: if None, will use default transforms
        subset_ids: select a subset of the full dataset
        strict_valid_test: replace validation split with test split (or vice versa)
                           if the dataset doesn't have a validation / test split
    """
    check_split(name, split, strict_valid_test)
    if name.lower() in ['cifar10', 'cifar-10']:
        from datasets.CIFAR10 import CIFAR10, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = CIFAR10(root=dataroot, train=(split == 'train'), transform=transforms)
    elif name.lower() in ['cifar100', 'cifar-100']:
        from datasets.CIFAR100 import CIFAR100, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = CIFAR100(root=dataroot, train=(split == 'train'), transform=transforms)
    else:
        raise ValueError(f"Dataset {name} is not supported.")

    if subset_ids is not None and len(subset_ids) < len(dataset):
        dataset = Subset(dataset, subset_ids)
    return dataset


def build_dataloader(dataset,
                     batch_size,
                     shuffle=False,
                     num_workers=0,
                     collate_fn=None,
                     pin_memory=False,
                     drop_last=False,
                     prefetch_factor=2):
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = None
    else:
        sampler = None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )
    return dataloader


def get_data_generator(dataloader, start_epoch=0):
    ep = start_epoch
    while True:
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(ep)
        for batch in dataloader:
            yield batch
        ep += 1


def _test_dist():
    import torch
    from utils.dist import init_distributed_mode, get_rank

    init_distributed_mode()
    assert is_dist_avail_and_initialized(), f'this test function only works in distributed mode'
    dataset = torch.arange(80)
    dataloader = build_dataloader(dataset, batch_size=4, shuffle=True)
    data_generator = get_data_generator(dataloader)
    f = open(f"./output{get_rank()}.txt", 'w')
    for i in range(100):
        batch = next(data_generator)
        print(get_rank(), i, batch, file=f)
    f.close()


if __name__ == '__main__':
    _test_dist()
