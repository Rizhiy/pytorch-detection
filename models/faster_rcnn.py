import torch
import torch.nn as nn
from torch.nn import DataParallel

from utils.config import cfg
from .feature_extractors import FeatureExtractor
from .rpn.rpn import RPN
from torch.autograd import Variable


class FasterRCNN(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rpn = RPN(feature_extractor.num_out_filters, feature_extractor.feature_stride)

    def forward(self, imgs, img_info, gt_boxes):
        img_info = img_info.data
        gt_boxes = gt_boxes.data
        features = self.feature_extractor(imgs)
        rois, rpn_cls_loss, rpn_box_loss = self.rpn(features, img_info, gt_boxes)
        return Variable(rois)
