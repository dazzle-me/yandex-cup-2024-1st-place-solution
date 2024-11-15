from typing import Dict
import torch
from torch import Tensor

def batch_to_device(batch: Dict[str, Tensor], device='cuda:0'):
    for k in batch.keys():
        if k not in ['track_id', 'clique']:
            batch[k] = batch[k].to(device, non_blocking=True)
    fts = batch['sample']
    fts[torch.isnan(fts)] = 0
    return batch

def describe_batch(batch: Dict):
    for k, v in batch.items():
        try:
            print(k, batch[k].shape)
        except:
            print(k, len(batch[k]))

def calc_tensor_stats(tensor: Tensor):
    print(
f'''
    mean : {tensor.mean()}
    std : {tensor.std()}
    min : {tensor.min()}
    max : {tensor.max()}
'''
)