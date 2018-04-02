import argparse
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataset
from dataset.collate import resize_collate
from models import create_feature_extractor
from models.faster_rcnn import FasterRCNN
from utils.config import cfg, update_config
from utils.serialisation import load_checkpoint, load_detections, save_detections


# TODO: Add logging


def test(batch_size: int = 1, workers: int = 4, cached: bool = False):
    """
    Test a network specified in global config
    :param batch_size: number nof images per GPU
    :param workers: number of workers
    :param cached: whether to use cached detection
    """
    test_imdb = create_dataset(cfg.DATASET.NAME, [cfg.DATASET.TEST_SET], cfg.DATASET.BASE_PATH, False)

    num_gpus = torch.cuda.device_count()
    # TODO: Work on multi-img test

    if cached:
        results = load_detections()
        bbox_pred, cls_prob = results['bbox_pred'], results['cls_prob']
    else:
        test_loader = DataLoader(test_imdb, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, pin_memory=True, collate_fn=resize_collate)

        feature_extractor = create_feature_extractor(cfg.NETWORK.FEATURE_EXTRACTOR.TYPE,
                                                     pretrained=cfg.NETWORK.FEATURE_EXTRACTOR.PRETRAINED,
                                                     depth=cfg.NETWORK.FEATURE_EXTRACTOR.DEPTH)
        net = FasterRCNN(feature_extractor, test_imdb.num_classes)

        if cfg.CUDA:
            net = DataParallel(net.cuda())

        net.load_state_dict(load_checkpoint(cfg.TEST.EPOCH)['weights'])

        print("Starting test")

        cls_prob = []
        bbox_pred = []

        for idx, (imgs, img_info, boxes) in enumerate(tqdm(test_loader)):

            input_imgs = Variable(imgs)
            input_info = Variable(img_info)
            input_boxes = Variable(boxes)

            if cfg.CUDA:
                input_imgs, input_info, input_boxes = input_imgs.cuda(), input_info.cuda(), input_boxes.cuda()

            batch_cls_prob, batch_bbox_pred, _ = net(input_imgs, input_info, input_boxes)
            for img_cls_prob, img_bbox_pred in zip(batch_cls_prob, batch_bbox_pred):
                img_cls_prob = img_cls_prob.data.cpu().numpy()
                img_bbox_pred = img_bbox_pred.data.cpu().numpy()
                cls_prob.append(img_cls_prob)
                bbox_pred.append(img_bbox_pred)
        save_detections({'bbox_pred': bbox_pred,
                         'cls_prob': cls_prob})

    result = test_imdb.evaluate(bbox_pred, cls_prob)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a network",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg', help="Path to the config to use", type=Path, metavar='PATH')
    parser.add_argument('-b', '--batch-size', help="Images per GPU", type=int, default=1)
    parser.add_argument('-w', '--workers', help="Number of workers", type=int, default=4)
    parser.add_argument('-c', '--cached', help="Whether to use cached detections", action='store_true')
    args = parser.parse_args()

    print(f"Called with args: {args}\n")

    update_config(args.cfg)

    if cfg.CUDA and not torch.cuda.is_available():
        print("PyTorch can't detect GPU, setting cfg.CUDA=False")
        cfg.CUDA = False

    test(args.batch_size, args.workers, args.cached)
