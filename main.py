import argparse
from configs.defaults import get_cfg_defaults

import engine


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


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = get_cfg_defaults()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    trainer = engine.Trainer(cfg, args)
    trainer.run_loop()
