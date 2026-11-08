"""Make a training run repeatable.

Without this, weight initialisation and batch order are drawn freshly every run, so two runs
with the same configuration are two independent samples and neither can be reproduced. That
makes it impossible to attribute a change in the metrics to a change in the code, and it means
a published checkpoint cannot be regenerated from the repository.

``apply_seed`` seeds every generator the training path draws from and returns a generator for
the data loader to use, so the batch order is fixed too.
"""

import os
import random

import numpy as np
import torch


def apply_seed(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch, and return a seeded generator for data loaders."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, not {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed a data-loader worker process from the seed torch assigned to it."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
