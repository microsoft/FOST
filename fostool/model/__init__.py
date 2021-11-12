# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
import hashlib
from .sandwich import SandwichModel
from .krnn import KRNNModel
from .mlp import MLP_Res

class TSModel(object):
    def __init__(self, model_name, model_params):
        #should not sent config directly
        self.model = eval(model_name)(**model_params)
        self.model_name = model_name
        self.model_params = model_params

    def load_weight(self, model_weight):
        self.model.load_state_dict(model_weight)

    def load_weight_from_path(self, path):
        store_dict = torch.load(path)
        if 'model' in store_dict:
            self.model.load_state_dict(store_dict['model'])

    def dump_weight(self, path, only_weight=True):
        if not only_weight:
            dump_dict = {
                'config': self.model_params,
                'name': self.name
            }
        else:
            dump_dict = {}
        if self.model is not None:
            dump_dict['model'] = self.model.state_dict()
        torch.save(dump_dict, path)

    def name(self):
        params_str = ''
        for k,v in sorted(self.model_params.items()):
            params_str += k
            params_str += str(v)
        _hash = hashlib.md5(params_str.encode(encoding='utf-8')).hexdigest()
        return self.model_name + '_' + str(_hash)[-4:]

    def __repr__(self):
        return self.model_name


