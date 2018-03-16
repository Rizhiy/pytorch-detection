# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
# Refactored by Artem Vasenin
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn

from utils.config import cfg
from utils.nms import nms
from .utils import generate_anchors, clip_boxes


class ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super().__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, scores, bbox_deltas, img_info, gt_boxes):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = scores[:, self._num_anchors:, :, :]
        cfg_key = 'TRAIN' if self.training else 'TEST'

        pre_nms_topN = cfg[cfg_key].RPN.NMS.PRE_TOP_N
        post_nms_topN = cfg[cfg_key].RPN.NMS.POST_TOP_N
        nms_thresh = cfg[cfg_key].RPN.NMS.THRESH

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = self.bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, img_info, batch_size)

        # TODO: We also supposed to filter boxes that are too small

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # 3. remove predicted boxes with either height or width < threshold
            # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single

        return output

    @staticmethod
    def bbox_transform_inv(boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0::4]
        dy = deltas[:, :, 1::4]
        dw = deltas[:, :, 2::4]
        dh = deltas[:, :, 3::4]

        pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
        pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
        pred_w = torch.exp(dw) * widths.unsqueeze(2)
        pred_h = torch.exp(dh) * heights.unsqueeze(2)

        pred_boxes = deltas.clone()
        # x1
        pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes
