from .feature_extractors import resnet


# TODO: Perhaps rename 'feature extractor' to 'backbone'
def create_feature_extractor(cls: str, pretrained=True, **kwargs):
    return eval(cls)(pretrained=pretrained, **kwargs)
