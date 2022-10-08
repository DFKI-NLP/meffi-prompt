import random
from typing import Dict, Any

import torch
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path


def seed_everything(seed: int) -> None:
    """Sets random seed anywhere randomness is involved.

    This process makes sure all the randomness-involved operations yield the
    same result under the same `seed`, so each experiment is reproducible.
    In this function, we set the same random seed for the following modules:
    `random`, `numpy` and `torch`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_relative_path(cfg: DictConfig) -> None:
    """Resolves all the relative path(s) given in `config.dataset` into absolute path(s).
    This function makes our code runnable in docker as well, where using relative path has
    problem with locating dataset files in `src/../data`.

    Args:
        cfg: Configuration of the experiment given in a dict.

    Example:
        Given `cfg.train_file="./data/smiler/train.json` and we call from
        "/netscratch/user/code/meffi-prompt/main.py", then `cfg.train_file` is
        overwritten by `/netscratch/user/code/meffi-prompt/data/smiler/train.json`.
    """
    for file_path in ["train_file", "eval_file", "test_file"]:
        if file_path in cfg:
            # enable simple path for smiler dataset splits, e.g. "en-train" or "de-test"
            if "-" in cfg[file_path]:
                splitted_path = cfg[file_path].split("-")
                cfg[file_path] = "./data/smiler/{}_corpora_{}.json".format(
                    splitted_path[0], splitted_path[1]
                )
            cfg[file_path] = to_absolute_path(cfg[file_path])


def aggregate_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate all the values of each column into a list of values.
    This step should be done during data-loading.
    """
    return {
        column_name: [example[column_name] for example in batch]
        for column_name in batch[0]
    }
