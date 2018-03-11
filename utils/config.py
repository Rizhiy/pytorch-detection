from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.name = ''

cfg.NETWORK = edict()
cfg.NETWORK.PIXEL_MEANS = (0.485, 0.456, 0.406)
cfg.NETWORK.PIXEL_STDS = (0.229, 0.224, 0.225)
cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.CACHE_FOLDER = 'cache'
cfg.DATASET.IMG_CHANNELS = 3

cfg.TRAIN = edict()

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

cfg.TEST = edict()

cfg.TEST.INPUT_SIZE = 600

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
                        default_dict[key] = tuple(value)
                    else:
                        default_dict[key] = value
            else:
                raise KeyError(f"key ({key}) must exist in config.py")

    with config_path.open() as config:
        exp_config = edict(yaml.load(config))
        parse_dict(cfg, exp_config, [])
