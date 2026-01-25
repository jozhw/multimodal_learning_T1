from pathlib import Path

import torch
import random
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate


def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)
    return [default_collate(samples) for samples in transposed]


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def monitor_memory(location=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # gb
        reserved = torch.cuda.memory_reserved() / (1024**3)  # gb
        print(
            f"[{location}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)  # obtain init from global that was set
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)  # If worker does any torch ops
