# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Refactored by Artem Vasenin
# --------------------------------------------------------

import numpy as np


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

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes
