# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
def decorate_batch(batch, device='cpu'):
    """Decorate the input batch with a proper device
    
    Parameters
    ----------
    batch : {[torch.Tensor | list | dict]}
        The input batch, where the list or dict can contain non-tensor objects
    device: str, optional
        'cpu' or 'cuda'

    Raises:
    ----------
        Exception: Unsupported data type

    Return
    ----------
    torch.Tensor | list | dict
        Maintain the same structure as the input batch, but with tensors moved to a proper device.
        
     """
    
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
        return batch
    elif isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict) or isinstance(value, list):
                batch[key] = decorate_batch(value, device)
            # retain other value types in the batch dict
        return batch
    elif isinstance(batch, list):
        new_batch = []
        for value in batch:
            if isinstance(value, torch.Tensor):
                new_batch.append(value.to(device))
            elif isinstance(value, dict) or isinstance(value, list):
                new_batch.append(decorate_batch(value, device))
            else:
                # retain other value types in the batch list
                new_batch.append(value)
        return new_batch
    else:
        raise Exception('Unsupported batch type {}'.format(type(batch)))