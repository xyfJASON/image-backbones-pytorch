from yacs.config import CfgNode as CN


_C = CN()
_C.SEED = 1234                          # random seed

_C.MODEL = CN()
_C.MODEL.NAME = 'resnet18'              # name of the model
_C.MODEL.FIRST_BLOCK = 'cifar10'        # only for models that require this argument. Options: 'cifar10' or 'imagenet'
_C.MODEL.PATCH_SIZE = 4                 # only for models that require this argument, typically Vision Transformers
_C.MODEL.WEIGHTS = None                 # path to the pretrained weights

_C.DATA = CN()
_C.DATA.NAME = 'CIFAR-10'               # name of the dataset
_C.DATA.DATAROOT = '/data/CIFAR-10/'    # path to the dataset
_C.DATA.IMG_SIZE = 32                   # size of the image
_C.DATA.N_CLASSES = 10                  # number of classes in the dataset

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4           # number of workers
_C.DATALOADER.PIN_MEMORY = True         # pin memory
_C.DATALOADER.PREFETCH_FACTOR = 2       # prefetch factor
_C.DATALOADER.BATCH_SIZE = 256          # batch size on each GPU
_C.DATALOADER.MICRO_BATCH = 0           # in case the GPU memory is too small, split a batch into micro batches
                                        # the gradients of micro batches will be aggregated for an update step

_C.TRAIN = CN()
_C.TRAIN.TRAIN_STEPS = 64000            # training duration, one step means one gradient update
_C.TRAIN.RESUME = None                  # options: path to checkpoint, 'best', 'latest', None
_C.TRAIN.PRINT_FREQ = 200               # frequency of printing status, in steps
_C.TRAIN.SAVE_FREQ = 5000               # frequency of saving checkpoints, in steps
_C.TRAIN.EVAL_FREQ = 1000               # frequency of evaluating the model, in steps
_C.TRAIN.USE_FP16 = False               # whether to use fp16

_C.TRAIN.OPTIM = CN()
_C.TRAIN.OPTIM.NAME = 'SGD'             # name of the optimizer
_C.TRAIN.OPTIM.LR = 0.1                 # learning rate
_C.TRAIN.OPTIM.WEIGHT_DECAY = 0.0005    # weight decay
_C.TRAIN.OPTIM.MOMENTUM = 0.9           # momentum, for SGD

_C.TRAIN.SCHED = CN()
_C.TRAIN.SCHED.NAME = 'CosineAnnealingLR'               # name of the scheduler
_C.TRAIN.SCHED.COSINE_T_MAX = 64000                     # argument of CosineAnnealingLR
_C.TRAIN.SCHED.COSINE_ETA_MIN = 0.001                   # argument of CosineAnnealingLR
_C.TRAIN.SCHED.MULTISTEP_MILESTONES = [32000, 48000]    # argument of MultiStepLR
_C.TRAIN.SCHED.MULTISTEP_GAMMA = 0.1                    # argument of MultiStepLR
_C.TRAIN.SCHED.WARMUP_STEPS = 0                         # warmup steps
_C.TRAIN.SCHED.WARMUP_FACTOR = 0.01                     # warmup factor


def get_cfg_defaults():
    return _C.clone()
