# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import numpy as np
import torch
from cpu_nms import cpu_nms

from utils.config import cfg
from .nms_gpu import nms_gpu


def nms(dets: torch.FloatTensor, thresh: float):
    if dets.shape[0] == 0:
        return []
    if cfg.CUDA:
        return nms_gpu(dets, thresh)
    else:
        np_dets = dets.numpy()
        np_nms = cpu_nms(np_dets, thresh)
        return torch.from_numpy(np.array(np_nms))
