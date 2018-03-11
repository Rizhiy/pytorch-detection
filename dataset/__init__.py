from pathlib import Path
from typing import List

from torch.utils.data import ConcatDataset

from .pascal_voc import PASCAL_VOC


def create_dataset(name: str, sets: List[str], base_path: Path, augment: bool):
    datasets = []
    for set_name in sets:
        datasets.append(eval(name)(set_name, base_path, augment=augment))
    # TODO: should ideally join them and re-sort by aspect ratio
    return ConcatDataset(datasets)
