from typing import Tuple

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from utils.config import cfg
from utils.net_utils import smooth_l1_loss
from .anchor_target_layer import AnchorTargetLayer
from .proposal_layer import ProposalLayer


class RPN(nn.Module):

    def __init__(self, filters_in: int, feat_stride: int):
        super().__init__()
        self.filters_in = filters_in
        self.feat_stride = feat_stride
        self.anchor_scales = cfg.NETWORK.RPN.ANCHOR_SCALES
        self.anchor_ratios = cfg.NETWORK.RPN.ANCHOR_RATIOS

        # One layer for additional features
        # TODO: experiment with different dilations
        self.conv = nn.Conv2d(self.filters_in, cfg.NETWORK.RPN.FILTERS, 3, padding=1)

        # Cls score for each anchor
        self.num_scores_out = self._num_anchors * 2  # Class agnostic for now
        self.cls_conv = nn.Conv2d(cfg.NETWORK.RPN.FILTERS, self.num_scores_out, 1, padding=1)

        # BBox offsets for each anchor
        self.num_bbox_out = self._num_anchors * 4
        self.bbox_conv = nn.Conv2d(cfg.NETWORK.RPN.FILTERS, self.num_bbox_out, 1, padding=1)

        self.proposal_layer = ProposalLayer(feat_stride, cfg.NETWORK.RPN.ANCHOR_SCALES, cfg.NETWORK.RPN.ANCHOR_RATIOS)
        self.anchor_target_layer = AnchorTargetLayer(feat_stride, cfg.NETWORK.RPN.ANCHOR_SCALES,
                                                     cfg.NETWORK.RPN.ANCHOR_RATIOS)

    @property
    def _num_anchors(self) -> int:
        return len(self.anchor_ratios) * len(self.anchor_scales)

    def forward(self, features, img_info, gt_boxes) -> Tuple[Variable, Tuple]:
        batch_size = features.size(0)

        conv = F.relu(self.conv(features))

        cls_score = self.cls_conv(conv)
        bbox_pred = self.bbox_conv(conv)

        # To calculate probabilities we need to reshape so that we can softmax
        shape = cls_score.size()
        cls_score_reshape = cls_score.view(shape[0], 2, -1, shape[3])
        cls_prob_reshape = F.softmax(cls_score_reshape, dim=1)
        cls_prob = cls_prob_reshape.view(shape[0], self.num_scores_out, -1, shape[3])

        rois = self.proposal_layer(cls_prob.data, bbox_pred.data, img_info, gt_boxes)

        cls_loss = 0.
        box_loss = 0.

        if self.training:
            labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = self.anchor_target_layer(
                cls_score.data, img_info, gt_boxes)

            # compute classification loss
            rpn_cls_score = cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = labels.view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            cls_loss = F.cross_entropy(rpn_cls_score, rpn_label, ignore_index=-1)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            # TODO: replace custom loss, with loss provided by PyTorch
            box_loss = smooth_l1_loss(bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                      rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

        return rois, (cls_loss, box_loss)
