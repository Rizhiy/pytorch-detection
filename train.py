import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from dataset.collate import resize_collate
from utils.config import update_config, cfg

parser = argparse.ArgumentParser(description="Train a network",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg', help="Path to the config to use", type=Path, metavar='PATH')
args = parser.parse_args()

update_config(args.cfg)

train_imdb = create_dataset(cfg.DATASET.NAME, cfg.DATASET.TRAIN_SETS, cfg.DATASET.BASE_PATH, cfg.DATASET.AUGMENT_TRAIN)

num_gpus = torch.cuda.device_count()
# train_loader = DataLoader(train_imdb, batch_size=num_gpus * cfg.TRAIN.BATCH_IMAGES, shuffle=False,
#                           num_workers=num_gpus * cfg.TRAIN.WORKERS, pin_memory=True, collate_fn=resize_collate)
train_loader = DataLoader(train_imdb, batch_size=2, shuffle=False,
                          num_workers=1, pin_memory=True, collate_fn=resize_collate)

print(next(iter(train_loader)))
