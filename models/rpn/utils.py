# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Refactored by Artem Vasenin
# --------------------------------------------------------

import numpy as np
import torch


def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    def width_height_centre(anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """

        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def make_anchors(ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """

        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    def make_scaled_anchors(anchor, scales):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """

        w, h, x_ctr, y_ctr = width_height_centre(anchor)
        ws = w * scales
        hs = h * scales
        anchors = make_anchors(ws, hs, x_ctr, y_ctr)
        return anchors

    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    # Make anchors with required aspect ratios with base area
    w, h, x_ctr, y_ctr = width_height_centre(base_anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    ratio_anchors = make_anchors(ws, hs, x_ctr, y_ctr)

    anchors = np.vstack([make_scaled_anchors(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def clip_boxes(boxes, im_shape, batch_size):
    # TODO: vectorise this
    # TODO: if not vectorise at least zip
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0., float(im_shape[i, 0] - 1))
        boxes[i, :, 1::4].clamp_(0., float(im_shape[i, 1] - 1))
        boxes[i, :, 2::4].clamp_(0., float(im_shape[i, 0] - 1))
        boxes[i, :, 3::4].clamp_(0., float(im_shape[i, 1] - 1))

    return boxes


def bbox_overlaps_batch(anchors: torch.FloatTensor, gt_boxes: torch.FloatTensor) -> torch.FloatTensor:
    # TODO: anchors should be (b, N, 4)
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (b, N, K) ndarray of overlap between boxes and query_boxes

    Where N is the number of anchors and K is max number of gt_boxes
    """
    if anchors.dim() != 3:
        raise ValueError('anchors input dimension is not correct.')

    batch_size = gt_boxes.size(0)

    N = anchors.size(1)
    K = gt_boxes.size(1)

    if anchors.size(2) == 4:
        anchors = anchors[:, :, :4].contiguous()
    else:
        anchors = anchors[:, :, 1:5].contiguous()

    gt_boxes = gt_boxes[:, :, :4].contiguous()

    gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
    gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

    anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
    anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
    anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
          torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
          torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)

    overlaps = iw * ih / ua

    # mask the overlap here.
    overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps


def bbox_transform_batch(pred_rois, gt_rois):
    assert pred_rois.dim() == 3, 'ex_roi input dimension is not correct.'

    pred_widths = pred_rois[:, :, 2] - pred_rois[:, :, 0] + 1.0
    pred_heights = pred_rois[:, :, 3] - pred_rois[:, :, 1] + 1.0
    pred_ctr_x = pred_rois[:, :, 0] + 0.5 * pred_widths
    pred_ctr_y = pred_rois[:, :, 1] + 0.5 * pred_heights

    gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
    gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
    gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - pred_ctr_x) / pred_widths
    targets_dy = (gt_ctr_y - pred_ctr_y) / pred_heights
    targets_dw = torch.log(gt_widths / pred_widths)
    targets_dh = torch.log(gt_heights / pred_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), 2)

    return targets


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

    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes
