# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.lib.npyio import _savez_compressed_dispatcher
import pandas as pd
import os
import logging
logger = logging.getLogger('Fusion')
logger.setLevel(logging.INFO)


class Fusion:
    """Fusing the results of multiple model predictions
    """
    def __init__(self, fusion_method='average'):
        """Fusion init

        Parameters
        ----------
        fusion_method : String, optional
            currently only support 'average' and 'trunacted'
        """
        assert fusion_method in ['average', 'trunacted'], \
            f'Unsupported fusion method {fusion_method}'
        self.fusion_method = fusion_method

    def __call__(self, result, loss_info=None):
        """Fusing the results of multiple model predictions

        Parameters
        ----------
        result : dict {String:DataFrame}
            {model name : predicted result of this model}
        loss_info : dict {String:float}, optional
            {model name : best validation loss score of this model}

        Returns
        -------
        pd.DataFrame
            fusion result
        """
        if self.fusion_method == 'average':
            stacked = []
            for _, v in result.items():
                stacked.append(v.values.reshape(-1, 1))
            stacked = np.concatenate(stacked, axis=-1)
            stacked = np.mean(stacked, axis=-1)

            return pd.DataFrame(stacked, index=v.index, columns=v.columns)

        if self.fusion_method == 'trunacted':
            stacked = []

            _model_name = []
            _loss_info_values = []
            for k in sorted(loss_info.keys()):
                _model_name.append(k)
                _loss_info_values.append(loss_info[k])
            scores = pd.DataFrame(
                {'val_loss': _loss_info_values, 'model_name': _model_name}
            )

            _bst, _mean, _std = \
                scores['val_loss'].min(), \
                scores['val_loss'].mean(), \
                scores['val_loss'].std()

            logger.info(scores[scores['val_loss'] < (_bst + _std)])

            scores = scores[scores['val_loss'] <= (_bst + _std)]
            for _model_name in scores['model_name'].values:
                v = result[_model_name]
                stacked.append(v.values.reshape(-1, 1))
            stacked = np.concatenate(stacked, axis=-1)
            stacked = np.mean(stacked, axis=-1)

            return pd.DataFrame(stacked, index=v.index, columns=v.columns)
