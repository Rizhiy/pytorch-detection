from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.config import cfg
from utils.net_utils import smooth_l1_loss
from .feature_extractors import FeatureExtractor
from .roi_pooling import RoIPooling
from .rpn.proposal_target_layer import ProposalTargetLayer
from .rpn.rpn import RPN


# TODO: Check how weights are initialised

class FasterRCNN(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, num_classes: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_refiner = feature_extractor.feature_refiner
        self.rpn = RPN(feature_extractor.num_out_filters, feature_extractor.feature_stride)
        self.proposal_target_layer = ProposalTargetLayer(num_classes)
        self.RoIPooling = RoIPooling(cfg.NETWORK.POOLING_SIZE, cfg.NETWORK.POOLING_SIZE,
                                     1 / feature_extractor.feature_stride)

        self.box_pred = nn.Linear(feature_extractor.num_out_filters * 2,
                                  (1 if cfg.NETWORK.CLASS_AGNOSTIC else num_classes) * 4)
        self.cls_pred = nn.Linear(feature_extractor.num_out_filters * 2, num_classes)

    def forward(self, imgs, img_info, gt_boxes) -> Tuple[Variable, Variable, Tuple]:
        batch_size = imgs.size(0)

        img_info = img_info.data
        gt_boxes = gt_boxes.data
        features = self.feature_extractor(imgs)
        rois, rpn_losses = self.rpn(features, img_info, gt_boxes)

        if self.training:
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = self.proposal_target_layer(rois, gt_boxes)

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None

        rois = Variable(rois)

        # Pool features for each roi
        pooled_feat = self.RoIPooling(features, rois.view(-1, 5))

        final_features = self._heavy_head(pooled_feat)

        box_pred = self.box_pred(final_features)
        if self.training and not cfg.NETWORK.CLASS_AGNOSTIC:
            # select column
            box_pred = box_pred.view(box_pred.size(0), int(box_pred.size(1) / 4), 4)
            box_pred = torch.gather(box_pred, 1,
                                    rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            box_pred = box_pred.squeeze(1)

        cls_pred = self.cls_pred(final_features)
        cls_prob = F.softmax(cls_pred, dim=1)

        # TODO: this is a bit ugly, find a better way
        cls_loss = 0.
        box_loss = 0.

        if self.training:
            # Classification Loss
            cls_loss = F.cross_entropy(cls_pred, rois_label, ignore_index=-1)
            # Regression Loss
            box_loss = smooth_l1_loss(box_pred, rois_target, rois_inside_ws, rois_outside_ws)
        # Map back to original images
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        box_pred = box_pred.view(batch_size, rois.size(1), -1)

        return cls_prob, box_pred, (cls_loss, box_loss, *rpn_losses)

    def _heavy_head(self, pooled_features):
        # This is not how it is implemented in paper
        # TODO: Check method described in paper
        return F.relu(self.feature_refiner(pooled_features).mean(3).mean(2))
