import sys
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm as original_tqdm

from utils.config import cfg


def tqdm(desc: str, total: int):
    return original_tqdm(total=total, unit='img', unit_scale=True, dynamic_ncols=True, desc=desc, file=sys.stdout,
                         smoothing=0.1)


def tensorToImages(imgs: torch.FloatTensor) -> List[Image.Image]:
    """
    Reverses normalisation of tensors and converts them back to PIL Images
    :param imgs: Batch of images must be in BCHW format
    :return:
    """
    result = []
    for img in imgs:
        img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
        img = img * cfg.NETWORK.PIXEL_STDS + cfg.NETWORK.PIXEL_MEANS
        img = Image.fromarray((img * 256).astype(np.uint8))
        result.append(img)
    return result


def drawBoxes(img: Image, boxes: np.ndarray, thresh=0.5):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        if box[-1] < thresh:
            continue
        box = box.astype(int)
        draw.rectangle(tuple(box[:4]), outline=(0, 255, 0))
    return img
