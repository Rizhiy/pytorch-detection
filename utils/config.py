from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.name = ''

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

cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.CACHE_FOLDER = 'cache'
cfg.DATASET.IMG_CHANNELS = 3
cfg.DATASET.TRAIN_SETS = ['train']
cfg.DATASET.TEST_SET = 'test'
cfg.DATASET.BASE_PATH = None
cfg.DATASET.AUGMENT_TRAIN = True

cfg.TRAIN = edict()

# Main seed, if None then chosen at random
cfg.TRAIN.SEED = None
# Images will be scaled to fill the area
cfg.TRAIN.MAX_AREA = 600 * 1000
# Images per GPU
cfg.TRAIN.BATCH_IMAGES = 2
# Workers per GPU
cfg.TRAIN.WORKERS = 2
# Range of scales to train on, remember that images will be rescaled down to fit in memory
cfg.TRAIN.RESIZE_SCALES = (0.5, 2.)
# Minimum relative size to crop
cfg.TRAIN.CROP_MIN_SCALE = 0.8

# NMS
cfg.TRAIN.RPN_NMS = edict()
cfg.TRAIN.RPN_NMS.PRE_TOP_N = 12000
cfg.TRAIN.RPN_NMS.THRESH = 0.7
cfg.TRAIN.RPN_NMS.POST_TOP_N = 2000

cfg.TEST = edict()

cfg.TEST.INPUT_SIZE = 600

# NMS
cfg.TEST.RPN_NMS = edict()
cfg.TEST.RPN_NMS.PRE_TOP_N = 12000
cfg.TEST.RPN_NMS.THRESH = 0.7
cfg.TEST.RPN_NMS.POST_TOP_N = 2000

np.random.seed(cfg.TRAIN.SEED)


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
                    if name == 'TRAIN.RESIZE_SCALES' or \
                            name == 'NETWORK.PIXEL_MEANS' or \
                            name == 'NETWORK.PIXEL_STDS':
                        value = tuple(value)
                    elif name == 'DATASET.BASE_PATH':
                        value = Path(value)
                    default_dict[key] = value
            else:
                raise KeyError(f"key ({key}) must exist in config.py")

    with config_path.open() as config:
        exp_config = edict(yaml.load(config))
        parse_dict(cfg, exp_config, [])
