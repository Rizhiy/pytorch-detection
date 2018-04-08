from pathlib import Path
from typing import List, Callable

from dataset.imdb import CombinedDataset, IMDB
from utils.config import cfg
from .pascal_voc import PASCAL_VOC


def create_dataset(name: str, sets: List[str], transform: Callable = None, sort=True, flip=False,
                   **kwargs) -> CombinedDataset:
    # TODO: Integrate flip
    datasets = []
    for set_name in sets:
        datasets.append(eval(name)(set_name, transform=transform, sort=sort, **kwargs))
    if flip:
        new_sets = []
        for set in datasets:
            new_sets.append(set.create_flipped())
        datasets += new_sets
    # TODO: should ideally join them and re-sort by aspect ratio
    return CombinedDataset(datasets, transform=transform, sort=sort)
