import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse
from PIL import Image
from torch.utils.data import Dataset

from utils.config import cfg


class IMDB(Dataset):
    def __init__(self, name: str, img_set: str, classes: List[str], augment=False):
        # TODO: add argument for sorting by aspect ratio
        """
        Base Image Database class
        :param name: name of the dataset
        :param img_set: name of the subset
        :param classes: list of names for each class in the dataset, without background
        :param augment: whether to augment the data
        """
        super().__init__()
        self.name = name
        self.img_set = img_set
        self.augment = augment
        self.classes = ['__background__', *classes]
        self.img_index = self._create_img_index()
        self._cache_path = Path(cfg.DATASET.CACHE_FOLDER) / (self.name + '_' + self.img_set + '.pkl')

        if self._cache_path.exists():
            with self._cache_path.open('rb') as cache_file:
                self._img_data = pickle.load(cache_file)
            print(f"Loaded {self.name} cache from {self._cache_path}")
        else:
            self._cache_path.parent.mkdir(exist_ok=True)
            self._img_data = self._create_img_data()
            with self._cache_path.open('wb') as cache_file:
                pickle.dump(self._img_data, cache_file)
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
        shape = np.array((tw, th))

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

        img_data['scale'] = 1.
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

    def evaluate(self, bbox_pred: List[np.array], cls_prob: List[np.array], **kwargs) -> float:
        # Evaluate calculated boxes using some kind of metric
        dets = []
        for img_bbox_pred, img_cls_prob in zip(bbox_pred, cls_prob):
            img_cls_prob = np.array([max(enumerate(list(prob)), key=lambda x: x[1])[0] for prob in img_cls_prob])
            dets.append(np.concatenate([img_bbox_pred, np.expand_dims(img_cls_prob, 1)], axis=1))
        return self.calculate_mAP(dets)

    def calculatePR(self, dets: List[np.array], cls: int, ovthresh: float = 0.5):
        # extract gt objects for this class
        gt_boxes = []
        npos = 0
        for img_data in self._img_data:
            img_boxes = [x[1] for x in zip(img_data['classes'], img_data['boxes']) if x[0] == cls]
            gt_boxes.append({'boxes': np.array(img_boxes),
                             'detected': [False] * len(img_boxes)})
            npos += len(img_boxes)

        # flatten_boxes
        img_idxs = []
        flat_boxes = []
        for img_idx, img_boxes in enumerate(dets):
            for box in img_boxes:
                if box[-1] == cls:
                    img_idxs.append(img_idx)
                    flat_boxes.append(box)

        # sort by confidence
        if flat_boxes:
            img_idxs, flat_boxes = zip(*sorted(zip(img_idxs, flat_boxes), key=lambda x: x[1][-1]))
        else:
            img_idxs, flat_boxes = [], []

        flat_boxes = np.array(flat_boxes)

        nd = len(flat_boxes)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # go down dets and mark TPs and FPs
        for idx, (img_idx, box) in enumerate(zip(img_idxs, flat_boxes)):
            ovmax = -np.inf

            gt = gt_boxes[img_idx]['boxes']

            if gt.size:
                ixmin = np.maximum(gt[:, 0], box[0])
                iymin = np.maximum(gt[:, 1], box[1])
                ixmax = np.minimum(gt[:, 2], box[2])
                iymax = np.minimum(gt[:, 3], box[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                       (gt[:, 2] - gt[:, 0] + 1.) *
                       (gt[:, 3] - gt[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh and not gt_boxes[img_idx]['det']:
                tp[idx] = 1.
                gt_boxes[img_idx]['detected'][jmax] = True
            else:
                fp[idx] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        return prec, rec

    def calculateAP(self, dets: List[np.array], cls: int) -> float:
        prec, rec = self.calculatePR(dets, cls)
        # correct AP calcuation
        # first append sentinel values at the end
        mrec = np.concatenate([[0.], rec, [1.0]])
        mpre = np.concatenate([[0.], prec, [0.0]])

        # compute the precision envelope
        # TODO: check that this is a valid operation
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points where recall changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum(\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return float(ap)

    def calculate_mAP(self, dets: List[np.array]) -> float:
        aps = []
        for idx, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            ap = self.calculateAP(dets, idx)
            aps.append(ap)

        mAP = np.mean(aps)
        return float(mAP)


class CombinedDataset(IMDB):
    def __init__(self, datasets: List[IMDB]):
        assert len(datasets) > 0, 'Datasets should not be an empty iterable'
        self.datasets = datasets
        super().__init__("CombinedDataset",
                         '+'.join([d.name + '_' + d.img_set for d in datasets]),
                         datasets[0].classes[1:])

        dclasses = [x.classes for x in datasets]
        assert dclasses.count(dclasses[0]) == len(dclasses), "Datasets have different classes!"

    def _create_img_data(self):
        img_data = []
        for dataset in self.datasets:
            for idx in dataset._index_map:
                img_data.append(dataset._img_data[idx])
        return img_data

    def _create_img_index(self):
        img_idxs = []
        for dataset in self.datasets:
            for idx in dataset._index_map:
                img_idxs.append(dataset.img_index[idx])
        return img_idxs
