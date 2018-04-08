import argparse
from pathlib import Path

from torchvision.transforms import ToPILImage
import torch
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import create_dataset
from dataset.collate import resize_collate
from dataset.transforms import DetCompose, DetRandomCrop, DetResize
from models import create_feature_extractor
from models.faster_rcnn import FasterRCNN
from test import test
from utils import tqdm, tensorToImage
from utils.config import cfg, update_config
from utils.serialisation import save_checkpoint, load_checkpoint, delete_detections

from tensorboardX import SummaryWriter
from datetime import datetime
from PIL import Image, ImageDraw

import numpy as np

import time


# TODO: Add logging

def train(batch_size: int = 1, workers: int = 4, resume: int = 0, validate=False, keep_det=False):
    # Check that we will be able to save the output
    """
    Train a network specified in global config
    :param batch_size: number of images per GPU
    :param workers: number of workers
    :param resume: epoch to resume from
    :param validate: whether to validate at the end of each epoch
    :param keep_det: whether to keep old detections after training has finished
    """
    save_checkpoint()

    # Add tensorboard for logging
    writer = SummaryWriter(f"runs/{cfg.NAME} {datetime.now().strftime('%m-%d %H:%M')}")

    transforms = []
    if cfg.TRAIN.CROP:
        transforms.append(DetRandomCrop(cfg.TRAIN.CROP_MIN_SCALE))
    if cfg.TRAIN.RESIZE:
        transforms.append(DetResize(*cfg.TRAIN.RESIZE_SCALES))

    train_transform = DetCompose(transforms)

    train_imdb = create_dataset(cfg.DATASET.NAME, cfg.DATASET.TRAIN_SETS,
                                transform=train_transform, sort=cfg.DATASET.SORT, flip=cfg.TRAIN.FLIP,
                                **cfg.DATASET.KWARGS)

    num_gpus = torch.cuda.device_count()
    batch_size *= num_gpus
    train_loader = DataLoader(train_imdb, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True, collate_fn=resize_collate(cfg.TRAIN.MAX_AREA))

    feature_extractor = create_feature_extractor(cfg.NETWORK.FEATURE_EXTRACTOR.TYPE,
                                                 pretrained=cfg.NETWORK.FEATURE_EXTRACTOR.PRETRAINED,
                                                 depth=cfg.NETWORK.FEATURE_EXTRACTOR.DEPTH)
    net = FasterRCNN(feature_extractor, train_imdb.num_classes)

    if resume:
        net.load_state_dict(load_checkpoint(resume)['weights'])

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

    # TODO: Try other optimisers
    optimiser = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # Reduce lr after few epoch
    scheduler = MultiStepLR(optimiser, milestones=cfg.TRAIN.STEPS, gamma=cfg.TRAIN.GAMMA)

    # TODO: find how to set current epoch in scheduler
    for i in range(resume):
        scheduler.step()

    for epoch in range(resume, cfg.TRAIN.EPOCHS):
        scheduler.step()
        offset = batch_size * len(train_loader) * epoch
        current_step = offset
        with tqdm(desc=f"Epoch {epoch}", total=len(train_loader.dataset)) as pbar:
            for idx, (imgs, img_info, boxes) in enumerate(train_loader):
                input_imgs = Variable(imgs)
                input_info = Variable(img_info)
                input_boxes = Variable(boxes)

                for img_idx, img in enumerate(tensorToImage(input_imgs.data)):
                    draw = ImageDraw.Draw(img)
                    boxes = input_boxes[img_idx].data.cpu().numpy()
                    for box in boxes:
                        box = box.astype(int)
                        draw.rectangle(tuple(box[:4]), outline=(0, 255, 0))
                    img.save(f"{batch_size*idx+img_idx}.jpg")
                if idx > 3:
                    exit()

                if cfg.CUDA:
                    input_imgs, input_info, input_boxes = input_imgs.cuda(), input_info.cuda(), input_boxes.cuda()

                _, _, losses = net(input_imgs, input_info, input_boxes)
                # Calculate loss
                # TODO: test sum()
                losses = [x.mean() for x in losses]
                head_cls_loss, head_box_loss = losses[0], losses[1]
                rpn_cls_loss, rpn_box_loss = losses[2], losses[3]

                loss = head_cls_loss + head_box_loss + rpn_cls_loss + rpn_box_loss

                # TODO: Check how to add multiple scalars at once
                current_step = batch_size * idx + offset
                writer.add_scalar('head_cls_loss', head_cls_loss, current_step)
                writer.add_scalar('head_box_loss', head_box_loss, current_step)
                writer.add_scalar('rpn_cls_loss', rpn_cls_loss, current_step)
                writer.add_scalar('rpn_box_loss', rpn_box_loss, current_step)
                writer.add_scalar('loss', loss, current_step)

                # Backprop
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Update progress bar
                pbar.update(batch_size)

        # TODO: Add optimiser state dict
        # TODO: Add summary writer name
        save_checkpoint({'epoch': epoch + 1,
                         'weights': net.module.state_dict()})
        if validate:
            del input_imgs, input_info, input_boxes, _, losses, \
                head_cls_loss, head_box_loss, rpn_cls_loss, rpn_box_loss, loss

            net.train(False)
            writer.add_scalar('validation', test([cfg.DATASET.VAL_SET], batch_size / (num_gpus * 2), workers, net=net),
                              current_step)
            net.train()
    if not keep_det:
        delete_detections()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a network",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg', help="Path to the config to use", type=Path, metavar='PATH')
    parser.add_argument('-b', '--batch-size', help="Images per GPU", type=int, default=1)
    parser.add_argument('-w', '--workers', help="Number of workers", type=int, default=4)
    parser.add_argument('-r', '--resume', help="Epoch to resume from", type=int, default=0)
    parser.add_argument('-v', '--validate', help='Whether to run validation after each epoch', action='store_true')
    parser.add_argument('-k', '--keep-det', help="Whether to keep old detections after training has finished",
                        action='store_true')
    args = parser.parse_args()

    print(f"Called with args: {args}\n")

    update_config(args.cfg)

    if cfg.CUDA and not torch.cuda.is_available():
        print("PyTorch can't detect GPU, setting cfg.CUDA=False")
        cfg.CUDA = False

    train(args.batch_size, args.workers, args.resume, args.validate, args.keep_det)
