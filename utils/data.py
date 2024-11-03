import tqdm
from torch.utils.data import Subset
from utils.logger import get_logger


def _check_split(name, split, strict_valid_test):
    available_split = {
        'cifar10': ['train', 'test'],
        'cifar-10': ['train', 'test'],
        'cifar100': ['train', 'test'],
        'cifar-100': ['train', 'test'],
    }
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


def get_dataset(name, dataroot, img_size, split, transforms=None, subset_ids=None, strict_valid_test=False):
    """
    Args:
        name: name of dataset
        dataroot: path to dataset
        img_size: size of images
        split: 'train', 'valid', 'test', 'all'
        transforms: if None, will use default transforms
        subset_ids: select a subset of the full dataset
        strict_valid_test: replace validation split with test split (or vice versa)
                           if the dataset doesn't have a validation / test split
    """
    split = _check_split(name, split, strict_valid_test)

    if name.lower() in ['cifar10', 'cifar-10']:
        from datasets.cifar10 import CIFAR10, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = CIFAR10(root=dataroot, train=(split == 'train'), transform=transforms)

    elif name.lower() in ['cifar100', 'cifar-100']:
        from datasets.cifar100 import CIFAR100, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = CIFAR100(root=dataroot, train=(split == 'train'), transform=transforms)

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    if subset_ids is not None and len(subset_ids) < len(dataset):
        dataset = Subset(dataset, subset_ids)
    return dataset


def _test_dist():
    import accelerate
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __init__(self, length: int):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, item):
            return item

    accelerator = accelerate.Accelerator()
    dataset = DummyDataset(80)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    dataloader = accelerator.prepare(dataloader)  # type: ignore
    data_generator = get_data_generator(dataloader, with_tqdm=False)
    f = open(f"./output{accelerator.process_index}.txt", 'w')
    for i in range(100):
        batch = next(data_generator)
        print(i, batch, file=f)
    f.close()


if __name__ == '__main__':
    _test_dist()
