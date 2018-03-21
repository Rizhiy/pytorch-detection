from pathlib import Path
from typing import List

from dataset.imdb import CombinedDataset, IMDB
from utils.config import cfg
from .pascal_voc import PASCAL_VOC


def create_dataset(name: str, sets: List[str], base_path: Path, augment: bool) -> CombinedDataset:
    datasets = []
    for set_name in sets:
        datasets.append(eval(name)(set_name, base_path, augment=augment))
    if cfg.TRAIN.FLIP:
        new_sets = []
        for set in datasets:
            new_sets.append(set.create_flipped())
        datasets += new_sets
    # TODO: should ideally join them and re-sort by aspect ratio
    return CombinedDataset(datasets)
