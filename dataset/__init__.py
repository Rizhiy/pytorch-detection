from typing import List

from .imdb import CombinedDataset, IMDB
from .pascal_voc import PASCAL_VOC
from .transforms import DetTransform


def create_dataset(name: str, sets: List[str], transform: DetTransform = None, sort=True,
                   **kwargs) -> CombinedDataset:
    datasets = []
    for set_name in sets:
        datasets.append(eval(name)(set_name, transform=transform, sort=sort, **kwargs))
    # TODO: should ideally join them and re-sort by aspect ratio
    return CombinedDataset(datasets, transform=transform, sort=sort)
