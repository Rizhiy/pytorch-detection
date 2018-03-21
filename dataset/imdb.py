import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from utils.config import cfg


class IMDB(Dataset):
    def __init__(self, name: str, img_set: str, classes: List[str], base_path: Path, augment=False):
        """
        Base Image Database class
        :param name: name of the dataset
        :param img_set: name of the subset
        :param classes: list of names for each class in the dataset, without background
        :param base_path: path to the root directory of the dataset
        :param augment: whether to augment the data
        """
        super().__init__()
        self.name = name
        self.img_set = img_set
        self.augment = augment
        self.classes = ['__background__', *classes]
        self._base_path = base_path
        self.img_index = self._create_img_index()
        self._cache_path = Path(cfg.DATASET.CACHE_FOLDER) / (self.name + '.pkl')

        if self._cache_path.exists():
            self._img_data = pickle.load(self._cache_path.open('rb'))
            print(f"Loaded {self.name} cache from {self._cache_path}")
        else:
            self._cache_path.parent.mkdir(exist_ok=True)
            self._img_data = self._create_img_data()
            pickle.dump(self._img_data, self._cache_path.open('wb'))
            print(f"Saved {self.name} cache to {self._cache_path}")
        self._index_map = self._create_sorted_index()

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @staticmethod
    def augmentation(img: Image, img_data: dict) -> Tuple[Image.Image, dict]:
        # TODO: This augmentation is a bit suboptimal, need to improve it
        shape = img_data['shape']
        crop_scale = np.random.uniform(low=cfg.TRAIN.CROP_MIN_SCALE)
        resize_scale = np.random.uniform(*cfg.TRAIN.RESIZE_SCALES)
        # First crop the image a bit
        w, h = shape
        tw, th = tuple((shape * crop_scale).astype(int))
        x = np.random.randint(0, w - tw)
        y = np.random.randint(0, h - th)
        img = img.crop((x, y, x + tw, y + th))
        # Then Scale
        target_shape = tuple((shape * resize_scale).astype(int))
        # Check if image will be large enough
        min_scale = min([x / cfg.NETWORK.MIN_SIZE for x in target_shape])
        if min_scale < 1:
            resize_scale /= min_scale
            target_shape = tuple((shape * resize_scale).astype(int))
        # TODO: check if this resizes correctly (no squeeze)
        img = img.resize(target_shape)

        # TODO: Check that all boxes are inside the crop/delete ones outside
        # Update boxes
        new_boxes = np.clip(img_data['boxes'] - [x, y, x, y], 0, np.inf) * resize_scale

        img_data.update({"scale": resize_scale,
                         "shape": target_shape,
                         "boxes": new_boxes})
        return img, img_data

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, idx) -> Tuple[Image.Image, dict]:
        """
        Return one image with corresponding data.
        Image should be PIL Image.
        Images will be normalised and converted to tensor during collation.
        Also to maintain maximum performance, images should be loaded in order (no shuffle)
        """
        true_idx = self._index_map[idx]
        img_path = self.img_index[true_idx]
        img_orig = Image.open(img_path)
        img_data = self._img_data[true_idx]

        if img_data['flipped']:
            img_orig = img_orig.transpose(Image.FLIP_LEFT_RIGHT)

        # create overlaps
        num_objs = len(img_data['classes'])
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        overlaps[np.arange(num_objs), img_data['classes']] = 1.
        overlaps = scipy.sparse.csr_matrix(overlaps)
        img_data['overlaps'] = overlaps

        assert (img_orig.size == img_data['shape']).all(), f"Image shape for {img_path} doesn't match stored shape"

        if self.augment:
            img, img_data = self.augmentation(img_orig, img_data)
        else:
            img, img_data = img_orig, img_data
        return img, img_data

    def _create_sorted_index(self):
        # shuffle img_index first, since sort is stable in python
        combined = list(zip(self.img_index, self._img_data))
        np.random.shuffle(combined)
        self.img_index[:], self._img_data[:] = zip(*combined)

        img_map = list(enumerate([data['shape'] for data in self._img_data]))
        img_map.sort(key=lambda x: x[1][0] / x[1][1])
        return [img_idx for img_idx, img_shape in img_map]

    def _create_img_index(self) -> List[Path]:
        "Should return a list of paths to each image in the dataset"
        raise NotImplementedError

    def _create_img_data(self) -> List[dict]:
        """
        Should return a list of dicts with each dict having the following items:
        'boxes': np.ndarray with shape (m,4) where m is the number of objects,
        they should have format of (x_min, y_min, x_max, y_max);
        'classes': np.ndarray with shape (m) where m is the number of objects;
        'shape': np.ndarray of shape (2) with contents being: (width, height);
        'flipped': whether the image should be flipped around horizontal axis
        """
        raise NotImplementedError

    def create_flipped(self) -> 'IMDB':
        new = deepcopy(self)
        for idx, data in enumerate(new._img_data):
            data['boxes'][:, [2, 0]] = data['shape'][0] - data['boxes'][:, [0, 2]]
            data['flipped'] = True
        return new


class CombinedDataset(ConcatDataset):
    def __init__(self, datasets: List[IMDB]):
        super().__init__(datasets)
        dclasses = [x.classes for x in datasets]
        assert dclasses.count(dclasses[0]) == len(dclasses), "Datasets have different classes!"
        self.num_classes = datasets[0].num_classes
