import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PASCAL_VOC
from dataset.collate import resize_collate
from utils.config import cfg, update_config

parser = argparse.ArgumentParser(description="Train a network",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg', help="Path to the config to use", type=Path, metavar='PATH')
args = parser.parse_args()

update_config(args.cfg)
print(cfg)
train_imdb = PASCAL_VOC('train', Path('data/VOCdevkit'), augment=True)

num_gpus = torch.cuda.device_count()
# train_loader = DataLoader(train_imdb, batch_size=num_gpus * cfg.TRAIN.BATCH_IMAGES, shuffle=False,
#                           num_workers=num_gpus * cfg.TRAIN.WORKERS, pin_memory=True, collate_fn=resize_collate)
train_loader = DataLoader(train_imdb, batch_size=2, shuffle=False,
                          num_workers=1, pin_memory=True, collate_fn=resize_collate)

print(next(iter(train_loader)))
