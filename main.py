import os
import argparse

import torch

from train_loop import TrainLoop
from utils.logger import get_logger
from utils.model import build_model
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.data import build_dataset, build_dataloader
from utils.misc import init_seeds, setup_cfg, create_exp_dir, get_time_str
from utils.dist import init_distributed_mode, get_world_size, get_rank, broadcast_objects


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name',
        help='name of experiment directory, if None, use current time instead',
    )
    parser.add_argument(
        '-c', '--config-file',
        metavar='FILE',
        help='path to config file',
    )
    parser.add_argument(
        '-ni', '--no-interaction',
        action='store_true',
        help='do not interacting with the user',
    )
    parser.add_argument(
        '--opts',
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line 'KEY VALUE' pairs",
    )
    return parser


def main():
    time_str = get_time_str()

    # PARSE ARGS & CONFIG
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    # INITIALIZE DISTRIBUTED MODE
    init_distributed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CREATE LOG DIRECTORY
    exp_dir = create_exp_dir(cfg, time_str, args.name, args.no_interaction)
    exp_dir = broadcast_objects(exp_dir)

    # INITIALIZE SEEDS
    init_seeds(cfg.SEED + get_rank(), cuda_deterministic=cfg.CUDA_DETERMINISTIC)

    # INITIALIZE LOGGER
    logger = get_logger(log_file=os.path.join(exp_dir, f'output-{time_str}.log'))
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Device: {device}')
    logger.info(f"Number of devices: {get_world_size()}")

    # BUILD DATASET & DATALOADER
    train_set = build_dataset(
        name=cfg.DATA.NAME,
        dataroot=cfg.DATA.DATAROOT,
        img_size=cfg.DATA.IMG_SIZE,
        split='train',
    )
    valid_set = build_dataset(
        name=cfg.DATA.NAME,
        dataroot=cfg.DATA.DATAROOT,
        img_size=cfg.DATA.IMG_SIZE,
        split='valid',
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Size of validation set: {len(valid_set)}')
    train_loader = build_dataloader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        prefetch_factor=cfg.DATALOADER.PREFETCH_FACTOR,
    )
    valid_loader = build_dataloader(
        dataset=valid_set,
        shuffle=False,
        drop_last=False,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        prefetch_factor=cfg.DATALOADER.PREFETCH_FACTOR,
    )
    effective_batch = cfg.DATALOADER.BATCH_SIZE * get_world_size()
    logger.info(f'Batch size per device: {cfg.DATALOADER.BATCH_SIZE}')
    logger.info(f"Effective batch size: {cfg.DATALOADER.BATCH_SIZE} * "
                f"{get_world_size()} == {effective_batch}")

    # BUILD MODEL & LOAD PRETRAINED WEIGHTS / RESUME
    model = build_model(
        name=cfg.MODEL.NAME,
        n_classes=cfg.DATA.N_CLASSES,
        img_size=cfg.DATA.IMG_SIZE,
    )
    if cfg.MODEL.WEIGHTS is not None:
        weights = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')
        model.load_state_dict(weights)
        logger.info(f'Successfully load model from {cfg.MODEL.WEIGHTS}')
    model.to(device)

    # BUILD OPTIMIZER & SCHEDULER
    params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = build_optimizer(params, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # START TRAINING
    train_loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        exp_dir=exp_dir,
        logger=logger,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        micro_batch=cfg.DATALOADER.MICRO_BATCH,
        train_steps=cfg.TRAIN.TRAIN_STEPS,
        resume=cfg.TRAIN.RESUME,
        print_freq=cfg.TRAIN.PRINT_FREQ,
        save_freq=cfg.TRAIN.SAVE_FREQ,
        eval_freq=cfg.TRAIN.EVAL_FREQ,
        use_fp16=cfg.TRAIN.USE_FP16,

        n_classes=cfg.DATA.N_CLASSES,
    )
    train_loop.run_loop()


if __name__ == '__main__':
    main()
