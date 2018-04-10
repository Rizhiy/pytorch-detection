from torch import nn
from torchvision.models.resnet import *


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.pretrained = pretrained

    @property
    def num_out_filters(self) -> int:
        """
        Should return number of filters in the output
        """
        raise NotImplementedError

    @property
    def feature_refiner(self) -> nn.Module:
        """
        Should return part of the network that can be used in the head of object detector
        """
        raise NotImplementedError

    @property
    def feature_stride(self) -> int:
        """
        Ratio of feature pixel size to original pixels
        """
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError


class resnet(FeatureExtractor):
    _factory = {18: resnet18,
                34: resnet34,
                50: resnet50,
                101: resnet101,
                152: resnet152}

    def __init__(self, depth: int, pretrained=True, **kwargs):
        super().__init__(pretrained, **kwargs)
        if depth not in self._factory:
            raise ValueError(f"Unsupported depth for resnet: {depth}. Options: {list(self._factory.keys())}")
        self.resnet = self._factory[depth](pretrained)
        self._num_out_filters = 512 * list(self.resnet.children())[-3][-1].expansion // 2
        self.features = nn.Sequential(*list(self.resnet.children())[:-3])
        # TODO: Modifify this to have dilation
        self._refiner = list(self.resnet.children())[-3]

    @property
    def num_out_filters(self) -> int:
        return self._num_out_filters

    @property
    def feature_refiner(self) -> nn.Module:
        return self._refiner

    @property
    def feature_stride(self):
        return 16

    def forward(self, input):
        return self.features(input)
