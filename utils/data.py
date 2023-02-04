from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.dist import is_dist_avail_and_initialized


def get_dataloader(dataset,
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
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=True)
    data_generator = get_data_generator(dataloader)
    f = open(f"./output{get_rank()}.txt", 'w')
    for i in range(100):
        batch = next(data_generator)
        print(get_rank(), i, batch, file=f)
    f.close()


if __name__ == '__main__':
    _test_dist()
