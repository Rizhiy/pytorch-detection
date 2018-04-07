import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch

from utils.config import cfg


def get_output_folder():
    output_folder = Path(cfg.OUTPUT) / cfg.NAME
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def save_checkpoint(state: dict = None):
    path = get_output_folder()
    if state is not None:
        path /= f"{state['epoch']}.checkpoint"
        torch.save(state, path)
        print(f"Saved weights to: {path}")


def load_checkpoint(epoch: int) -> dict:
    path = get_output_folder() / f"{epoch}.checkpoint"
    result = torch.load(path)
    print(f"Loaded weights from: {path}")
    return result


results_filename = "results.pkl"


# These functions are useful when developing evaluation metrics
def save_detections(results: dict):
    path = get_output_folder() / results_filename
    with path.open("wb") as results_file:
        pickle.dump(results, results_file)
    print(f"Saved detections to {path}")


def load_detections() -> dict:
    path = get_output_folder() / results_filename
    with path.open("rb") as results_file:
        results = pickle.load(results_file)
    print(f"Loaded detections from {path}")
    return results


def delete_detections():
    path = get_output_folder() / results_filename
    path.unlink()
    print("Deleted old detections")
