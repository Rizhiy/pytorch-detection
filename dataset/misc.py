from pathlib import Path

import numpy as np
import scipy.sparse


class ImgData:
    def __init__(self, path: Path, boxes: np.ndarray, classes: np.ndarray, shape: np.ndarray, num_classes: int):
        """
        :param path: path to image location
        :param boxes: np.ndarray with shape (m,4) where m is the number of objects,
        they should have format of (x_min, y_min, x_max, y_max);
        :param classes: np.ndarray with shape (m) where m is the number of objects;
        :param shape: np.ndarray of shape (2) with contents being: (width, height);
        """
        self.path = path
        self.boxes = boxes
        self.classes = classes
        self.shape = shape
        self.num_classes = num_classes

        self.scale = np.array([1.])
        self.flipped = False

    @property
    def num_objs(self) -> int:
        return len(self.classes)

    @property
    def overlaps(self) -> scipy.sparse.csr_matrix:
        overlaps = np.zeros((self.num_objs, self.num_classes), dtype=np.float32)
        overlaps[np.arange(self.num_objs), self.classes] = 1.
        return scipy.sparse.csr_matrix(overlaps)
