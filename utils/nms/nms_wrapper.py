# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from .nms_gpu import nms_gpu


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    return nms_gpu(dets, thresh)
