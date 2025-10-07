from pathlib import Path

import torch
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate


def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)
    return [default_collate(samples) for samples in transposed]


def _beta_mixed_collate(batch):
    if len(batch) == 0:
        return batch

    # Unpack the batch
    tcga_ids, days_to_events, events_occurred, wsi_data_list, omic_data = zip(*batch)

    # Handle each component appropriately
    tcga_ids = list(tcga_ids)  # Keep as list of strings

    # Convert scalar values to tensors
    days_to_events = torch.stack(
        [torch.tensor(d, dtype=torch.long) for d in days_to_events]
    )
    events_occurred = torch.stack(
        [torch.tensor(e, dtype=torch.long) for e in events_occurred]
    )

    # Handle WSI data: keep as list of lists (each inner list contains tiles for one patient)
    wsi_data = list(
        wsi_data_list
    )  # This preserves the structure: [patient1_tiles, patient2_tiles, ...]

    # Stack omic data normally
    omic_data = torch.stack(omic_data)

    return tcga_ids, days_to_events, events_occurred, wsi_data, omic_data


def monitor_memory(location=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # gb
        reserved = torch.cuda.memory_reserved() / (1024**3)  # gb
        print(
            f"[{location}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        )


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
