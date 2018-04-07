import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from .imdb import IMDB


class PASCAL_VOC(IMDB):
    obj_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, img_set: str, base_path: str, year: int, **kwargs):
        self._base_path = Path(base_path) / f"VOC{year}"
        super().__init__("pascal_voc", img_set, self.obj_classes, **kwargs)
        self.year = year

    def _create_img_index(self):
        with (self._base_path / "ImageSets" / "Main" / (self.img_set + '.txt')).open('r') as img_list:
            return [self._base_path / "JPEGImages" / (x.strip() + '.jpg') for x in img_list.readlines()]

    def _create_img_data(self):
        def load_pascal_annotation(path: Path):
            # --------------------------------------------------------
            # Fast R-CNN
            # Copyright (c) 2015 Microsoft
            # Licensed under The MIT License [see LICENSE for details]
            # Written by Ross Girshick
            # --------------------------------------------------------
            tree = ET.parse(path)
            objs = tree.findall('object')
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            # "Seg" area for pascal is just the box area
            seg_areas = np.zeros((num_objs), dtype=np.float32)
            ishards = np.zeros((num_objs), dtype=np.int32)

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                diffc = obj.find('difficult')
                difficult = 0 if diffc is None else int(diffc.text)
                ishards[ix] = difficult

                class_to_idx = dict(zip(self.classes, range(self.num_classes)))
                cls = class_to_idx[obj.find('name').text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            return {'boxes': boxes,
                    'classes': gt_classes,
                    'flipped': False,
                    'gt_ishard': ishards,
                    'seg_areas': seg_areas}

        with (self._base_path / "ImageSets" / "Main" / (self.img_set + '.txt')).open('r') as img_list:
            annotations_paths = [self._base_path / "Annotations" / (x.strip() + '.xml') for x in img_list.readlines()]
        img_data = [load_pascal_annotation(path) for path in annotations_paths]
        shapes = [np.array(Image.open(img_path).size) for img_path in self.img_index]

        for data, shape in zip(img_data, shapes):
            data.update({"shape": shape})

        return img_data
