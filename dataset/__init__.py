from pathlib import Path
from typing import List

from dataset.imdb import CombinedDataset, IMDB
from utils.config import cfg
from .pascal_voc import PASCAL_VOC


def create_dataset(name: str, sets: List[str], augment=False, sort=True, flip=False, **kwargs) -> CombinedDataset:
    # TODO: Integrate flip
    datasets = []
    for set_name in sets:
        datasets.append(eval(name)(set_name, augment=augment, sort=sort, **kwargs))
    if flip:
        new_sets = []
        for set in datasets:
            new_sets.append(set.create_flipped())
        datasets += new_sets
    # TODO: should ideally join them and re-sort by aspect ratio
    return CombinedDataset(datasets, augment=augment, sort=sort)
