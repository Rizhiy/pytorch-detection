import argparse
import time
from pathlib import Path

import torch
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataset
from dataset.collate import resize_collate
from models import create_feature_extractor
from models.faster_rcnn import FasterRCNN
from utils.config import cfg, update_config


# TODO: Add logging


def train(batch_images: int):
    train_imdb = create_dataset(cfg.DATASET.NAME, cfg.DATASET.TRAIN_SETS, cfg.DATASET.BASE_PATH,
                                cfg.DATASET.AUGMENT_TRAIN)

    num_gpus = torch.cuda.device_count()
    train_loader = DataLoader(train_imdb, batch_size=num_gpus * batch_images, shuffle=False,
                              num_workers=num_gpus * cfg.TRAIN.WORKERS, pin_memory=True, collate_fn=resize_collate)

    feature_extractor = create_feature_extractor(cfg.NETWORK.FEATURE_EXTRACTOR.TYPE,
                                                 pretrained=cfg.NETWORK.FEATURE_EXTRACTOR.PRETRAINED,
                                                 depth=cfg.NETWORK.FEATURE_EXTRACTOR.DEPTH)
    net = FasterRCNN(feature_extractor, train_imdb.num_classes)

    net.train()

    if cfg.CUDA:
        net = DataParallel(net.cuda())

    # Set up optimiser
    params = []
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            params.append({'params': [parameter],
                           'lr': cfg.TRAIN.LEARNING_RATE,
                           'weight_decay': 0 if 'bias' in name else cfg.TRAIN.WEIGHT_DECAY})
    optimiser = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # Reduce lr on after few epoch
    scheduler = MultiStepLR(optimiser, milestones=[4, 6], gamma=0.1)
    # TODO: Add resume mechanic

    avg_loss = 2
    for epoch in range(cfg.TRAIN.EPOCHS):
        scheduler.step()
        print(f"Starting epoch {epoch}")
        for idx, (imgs, img_info, boxes) in enumerate(tqdm(train_loader)):

            input_imgs = Variable(imgs)
            input_info = Variable(img_info)
            input_boxes = Variable(boxes)

            if cfg.CUDA:
                input_imgs, input_info, input_boxes = input_imgs.cuda(), input_info.cuda(), input_boxes.cuda()

            _, _, losses = net(input_imgs, input_info, input_boxes)
            # Calculate loss
            # TODO: test sum()
            loss = losses[0].mean() + losses[1].mean() + losses[2].mean() + losses[3].mean()
            avg_loss = avg_loss * 0.99 + loss.data[0] * 0.01

            # Backprop
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"Epoch={epoch}, Loss={avg_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a network",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg', help="Path to the config to use", type=Path, metavar='PATH')
    parser.add_argument('-b', help="Images per GPU", type=int, default=1)
    args = parser.parse_args()

    print(f"Called with args: {args}\n")

    update_config(args.cfg)

    if cfg.CUDA and not torch.cuda.is_available():
        print("PyTorch can't detect GPU, setting cfg.CUDA=False")
        cfg.CUDA = False

    train(args.b)
