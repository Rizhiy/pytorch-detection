from .feature_extractors import resnet


def create_feature_extractor(type: str, pretrained=True, **kwargs):
    return eval(type)(pretrained=pretrained, **kwargs)
