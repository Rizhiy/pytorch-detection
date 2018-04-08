from collections import namedtuple
from pathlib import Path
from typing import List

import numpy as np
import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.NAME = ''

cfg.CUDA = True
# Output folder, training data and logs will be placed in cfg.OUTPUT/cfg.name/<files>
cfg.OUTPUT = 'output'

# Parameters for network architecture
cfg.NETWORK = edict()
cfg.NETWORK.PIXEL_MEANS = (0.485, 0.456, 0.406)
cfg.NETWORK.PIXEL_STDS = (0.229, 0.224, 0.225)
# Parameters for feature extractor, not every feature extractor uses all of the parameters
cfg.NETWORK.FEATURE_EXTRACTOR = edict()
cfg.NETWORK.FEATURE_EXTRACTOR.TYPE = 'resnet'
cfg.NETWORK.FEATURE_EXTRACTOR.DEPTH = 50
cfg.NETWORK.FEATURE_EXTRACTOR.PRETRAINED = True

# RPN settings
cfg.NETWORK.RPN = edict()
cfg.NETWORK.RPN.ANCHOR_SCALES = (8, 16, 32)
cfg.NETWORK.RPN.ANCHOR_RATIOS = (0.5, 1.0, 2.0)
# Number of filters to use in the intermediate layer
cfg.NETWORK.RPN.FILTERS = 512
# Minimum image size, this should be slightly larger than the size of the smallest anchor in pixels
# TODO: make this a parameter in network
cfg.NETWORK.MIN_SIZE = cfg.NETWORK.RPN.ANCHOR_SCALES[0] * (16 + 1)
# RoIPooling
cfg.NETWORK.POOLING_SIZE = 7
# Whether bbox prediction is class agnostic
cfg.NETWORK.CLASS_AGNOSTIC = True
# Normalize the targets using "precomputed" (or made up) means and stdevs
cfg.NETWORK.TARGETS_NORMALISE = edict()
cfg.NETWORK.TARGETS_NORMALISE.PRECOMPUTED = True
cfg.NETWORK.TARGETS_NORMALISE.MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.NETWORK.TARGETS_NORMALISE.STDS = (0.1, 0.1, 0.2, 0.2)

cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.CACHE_FOLDER = 'cache'
cfg.DATASET.IMG_CHANNELS = 3
cfg.DATASET.TRAIN_SETS = ['train']
# TODO: Maybe allow testing on multiple sets
cfg.DATASET.VAL_SET = 'val'
cfg.DATASET.TEST_SET = 'test'
cfg.DATASET.SORT = True
# Key-word arguments specific to each dataset, must be a one level dict
cfg.DATASET.KWARGS = edict()

cfg.TRAIN = edict()

# Main seed, if None then chosen at random
cfg.TRAIN.SEED = None
# Number of epochs to train
cfg.TRAIN.EPOCHS = 1
# Optimiser settings
cfg.TRAIN.LEARNING_RATE = 0.01
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
# LR Scheduler parameters
cfg.TRAIN.STEPS = [5, ]
cfg.TRAIN.GAMMA = 0.1
# Transforms
cfg.TRAIN.CROP = True
cfg.TRAIN.RESIZE = True
cfg.TRAIN.FLIP = True
# Images will be scaled to fill the area
cfg.TRAIN.MAX_AREA = 600_000
# Images per GPU
cfg.TRAIN.BATCH_IMAGES = 2
# Range of scales to train on, remember that images will be rescaled down to fit in memory
# This range should ideally be geometrically centred at the TEST.RESIZE_SCALE
cfg.TRAIN.RESIZE_SCALES = (.5, 2.)
# Minimum relative size to crop
cfg.TRAIN.CROP_MIN_SCALE = 0.5
# If TRAIN.FLIP then images will be flipped with this probability
cfg.TRAIN.FLIP_PROB = 0.5
# Minibatch size (number of regions of interest [ROIs] to backprop) per image
cfg.TRAIN.BATCH_SIZE = 128
# Fraction of minibatch that is labeled foreground (i.e. class > 0)
cfg.TRAIN.FG_FRACTION = 0.25
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
cfg.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
cfg.TRAIN.BG_THRESH_HI = 0.5
cfg.TRAIN.BG_THRESH_LO = 0.0

cfg.TRAIN.RPN = edict()
# NMS
cfg.TRAIN.RPN.NMS = edict()
cfg.TRAIN.RPN.NMS.PRE_TOP_N = 12000
cfg.TRAIN.RPN.NMS.THRESH = 0.7
cfg.TRAIN.RPN.NMS.POST_TOP_N = 2000

# RPN training parameters
cfg.TRAIN.RPN.NEGATIVE_OVERLAP = 0.3
cfg.TRAIN.RPN.POSITIVE_OVERLAP = 0.7
cfg.TRAIN.RPN.CLOBBER_POSITIVES = False
cfg.TRAIN.RPN.BATCH_SIZE = 256
cfg.TRAIN.RPN.FG_FRACTION = 0.5
cfg.TRAIN.RPN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
cfg.TRAIN.RPN.POSITIVE_WEIGHT = -1

# Proposal Target Layer Params
cfg.TRAIN.BBOX = edict()
# TODO: Remove this
# Deprecated (inside weights)
cfg.TRAIN.BBOX.INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

cfg.TEST = edict()

cfg.TEST.EPOCH = cfg.TRAIN.EPOCHS

cfg.TEST.RESIZE_SCALE = 1.
cfg.TEST.MAX_AREA = 1_200_000  # Can be about double of TEST.MAX_AREA, since we don't store gradients during test
# TODO: Add default input size option to allow testing similar to competitions

cfg.TEST.RPN = edict()
# NMS
cfg.TEST.RPN.NMS = edict()
cfg.TEST.RPN.NMS.PRE_TOP_N = 12000
cfg.TEST.RPN.NMS.THRESH = 0.7
cfg.TEST.RPN.NMS.POST_TOP_N = 2000

np.random.seed(cfg.TRAIN.SEED)

test = namedtuple('test', ('a',))


def _get_tuples(d: dict) -> List[str]:
    results = []
    for k, v in d.items():
        if isinstance(v, tuple):
            results.append(k)
        if isinstance(v, dict):
            results += [k + '.' + x for x in _get_tuples(v)]
    return results


tuple_keys = _get_tuples(cfg)


def update_config(config_path: Path):
    def parse_dict(default_dict: dict, new_dict: dict, prefix):
        for key, value in new_dict.items():
            if key in default_dict:
                new_prefix = [*prefix, key]
                if isinstance(value, dict):
                    parse_dict(default_dict[key], new_dict[key], new_prefix)
                else:
                    # List special cases here
                    name = '.'.join(new_prefix)
                    if name in tuple_keys:
                        value = tuple(value)
                    elif name == 'DATASET.BASE_PATH':
                        value = Path(value)
                    default_dict[key] = value
            elif prefix[-1] == 'KWARGS':
                default_dict[key] = value
            else:
                print(prefix)
                raise KeyError(f"key ({key}) must exist in config.py")

    with config_path.open() as config:
        exp_config = edict(yaml.load(config))
        parse_dict(cfg, exp_config, [])
    np.random.seed(cfg.TRAIN.SEED)
