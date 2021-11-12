# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import os
import torch
import numpy as np
from .utils import decorate_batch
import pandas as pd

class Predictor(object):
    def __init__(self, device='cuda') -> None:
        """
        Parameters
        ----------
        device : str, optional
            'cpu' or 'cuda'
        """
        super().__init__()

        if device == 'cuda':
            assert torch.cuda.is_available(), 'No cuda device found.'

        self.device = device

    def __call__(self, test_data_loader, model, target_col):
        """
        Parameters
        ----------
        test_data_loader : DataLoader(pytorch data loader)
            data loader for testing data
        model: TSModel
            model for inference
        target_col: str
            Target column name
        
        Return
        ----------
        pred: DataFrame
            ------------------------
            | Node | Date | Target |
        """
        pred = []

        torch_model = model.model
        torch_model.to(self.device)

        for batch in test_data_loader:
            (x, y), g, _ = decorate_batch(batch, self.device)
            y_hat, _ = torch_model(x, g)

            cent_n_id = g['cent_n_id']
            res_n_id = g['res_n_id']
            y_hat = y_hat[:, res_n_id]
            cent_n_id = cent_n_id[res_n_id]

            index_ptr = torch.cartesian_prod(
                torch.arange(cent_n_id.size(0)),
                torch.arange(y.size(-1))
            )

            pred.append(
                pd.DataFrame({
                    'Node': cent_n_id[index_ptr[:, 0]].data.cpu().numpy(),
                    'Date': index_ptr[:, 1].data.cpu().numpy(),
                    target_col: y_hat.flatten().data.cpu().numpy()
                })
            )

        pred = pd.concat(pred, axis=0)
        return pred
