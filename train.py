import argparse
from pathlib import Path

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataset
from dataset.collate import resize_collate
from models import create_feature_extractor
from utils.config import update_config, cfg

parser = argparse.ArgumentParser(description="Train a network",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg', help="Path to the config to use", type=Path, metavar='PATH')
args = parser.parse_args()

update_config(args.cfg)

train_imdb = create_dataset(cfg.DATASET.NAME, cfg.DATASET.TRAIN_SETS, cfg.DATASET.BASE_PATH, cfg.DATASET.AUGMENT_TRAIN)

num_gpus = torch.cuda.device_count()
train_loader = DataLoader(train_imdb, batch_size=num_gpus * cfg.TRAIN.BATCH_IMAGES, shuffle=False,
                          num_workers=num_gpus * cfg.TRAIN.WORKERS, pin_memory=True, collate_fn=resize_collate)

feature_extractor = create_feature_extractor(cfg.NETWORK.FEATURE_EXTRACTOR.TYPE,
                                             pretrained=cfg.NETWORK.FEATURE_EXTRACTOR.PRETRAINED,
                                             depth=cfg.NETWORK.FEATURE_EXTRACTOR.DEPTH).cuda()
feature_refiner = nn.DataParallel(feature_extractor.feature_refiner.cuda())
feature_extractor = nn.DataParallel(feature_extractor)

for imgs, data in tqdm(train_loader):
    features = feature_extractor(Variable(imgs, volatile=True).cuda())
    output = feature_refiner(features)
