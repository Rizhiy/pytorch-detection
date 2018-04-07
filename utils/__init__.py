import sys
from typing import List

import torch
from PIL import Image
from tqdm import tqdm as original_tqdm

from utils.config import cfg
import numpy as np


def tqdm(desc: str, total: int):
    return original_tqdm(total=total, unit='img', unit_scale=True, dynamic_ncols=True, desc=desc, file=sys.stdout,
                         smoothing=0.1)


def tensorToImage(imgs: torch.FloatTensor) -> List[Image.Image]:
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
