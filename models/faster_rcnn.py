import torch.nn as nn
from torch.autograd import Variable

from .feature_extractors import FeatureExtractor
from .rpn.proposal_target_layer import ProposalTargetLayer
from .rpn.rpn import RPN


class FasterRCNN(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, num_classes: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.rpn = RPN(feature_extractor.num_out_filters, feature_extractor.feature_stride)
        self.proposal_target_layer = ProposalTargetLayer(num_classes)

    def forward(self, imgs, img_info, gt_boxes):
        img_info = img_info.data
        gt_boxes = gt_boxes.data
        features = self.feature_extractor(imgs)
        rois, rpn_cls_loss, rpn_box_loss = self.rpn(features, img_info, gt_boxes)

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
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        return rois
