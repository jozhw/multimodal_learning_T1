from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate

def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    transposed = zip(*batch)
    return [default_collate(samples) for samples in transposed]
