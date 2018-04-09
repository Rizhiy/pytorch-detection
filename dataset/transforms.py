from typing import Tuple, List

import numpy as np
from PIL.Image import Image, FLIP_LEFT_RIGHT

from dataset.misc import ImgData
from utils import cfg


class DetTransform:
    def __call__(self, img: Image, img_data: ImgData) -> Tuple[Image, ImgData]:
        raise NotImplementedError


def check_min_size(shape: Tuple[int, int]) -> np.ndarray:
    """
    Check that image shape is large enough for the network
    :param shape: current image shape
    :return: required scale factor
    """
    min_scale = min([x / cfg.NETWORK.MIN_SIZE for x in shape])
    resize_scale = np.array([1.])
    if min_scale < 1:
        resize_scale /= min_scale
    return resize_scale


# TODO: Perhaps replace these transformations with functions
class DetRandomCrop(DetTransform):
    def __init__(self, min_scale: float):
        self.min_scale = min_scale

    def __call__(self, orig_img: Image, img_data: ImgData) -> Tuple[Image, ImgData]:
        orig_shape = img_data.shape
        crop_scale = np.random.uniform(low=self.min_scale)

        w, h = orig_shape
        tw, th = tuple((orig_shape * crop_scale).astype(int))
        x = np.random.randint(0, w - tw)
        y = np.random.randint(0, h - th)

        new_img = orig_img.crop((x, y, x + tw, y + th))
        img_data.shape = np.array((tw, th))
        # TODO: Perhaps delete boxes that are too small
        img_data.boxes = np.clip(img_data.boxes - [x, y, x, y], 0, [tw, th, tw, th])

        return new_img, img_data


def resize(img: Image, img_data: ImgData, scale: np.ndarray) -> Tuple[Image, ImgData]:
    target_shape = tuple((img_data.shape * scale).astype(int))

    img = img.resize(target_shape)
    img_data.shape = target_shape
    img_data.boxes *= scale

    return img, img_data


class DetResize(DetTransform):
    def __init__(self, low: float, high: float = None):
        """
        Resize img and bounding boxes
        If only low is given, then will resize to value of low
        If both low and high are given, will resize to random value between the two
        :param low:
        :param high:
        """
        self.low = low
        self.high = high

    def __call__(self, orig_img: Image, img_data: ImgData) -> Tuple[Image, ImgData]:
        shape = img_data.shape
        if self.high is None:
            scale = self.low
        else:
            scale = np.random.uniform(self.low, self.high)

        additional_scale = check_min_size(tuple((shape * scale).astype(int)))
        scale *= additional_scale

        return resize(orig_img, img_data, scale)


class DetCompose(DetTransform):
    def __init__(self, transforms: List[DetTransform]):
        self.transforms = transforms

    def __call__(self, img: Image, img_data: ImgData) -> Tuple[Image, ImgData]:
        for t in self.transforms:
            img, img_data = t(img, img_data)
        return img, img_data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class DetFlip(DetTransform):
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, img: Image, img_data: ImgData) -> Tuple[Image, ImgData]:
        if np.random.uniform() < self.prob:
            img = img.transpose(FLIP_LEFT_RIGHT)
            img_data.boxes[:, [2, 0]] = img_data.shape[0] - img_data.boxes[:, [0, 2]]
            img_data.flipped = True
        return img, img_data
