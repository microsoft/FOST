# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_fixer import InputFixer


class MLP_Res(nn.Module):
    def __init__(self, num_nodes, fea_dim, hid_dim, lookback, lookahead, nb_layer, **params):
        """Build an MLP model

        Parameters
        ----------
        num_nodes : int
            The number of nodes
        fea_dim : int
            The feature dimension
        hid_dim : int
            The hidden dimension
        lookback : int
            The number of lookback steps
        lookahead : int
            The number of lookahead steps
        nb_layer : int
            The number of layers
        """
        super().__init__()

        self.input_fixer = InputFixer(
            num_nodes=num_nodes,
            fea_dim=fea_dim
        )

        self.input_fc = nn.Linear(lookback, hid_dim)
        self.res_layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(nb_layer)]
        )
        self.project = nn.Linear(hid_dim, lookahead)

    def forward(self, x, g=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        g : dict
            Graph sample

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            forecasts, updated inputs
        """
        x = self.input_fixer(x)

        x_fix = x

        sz = x.size()
        x = x.view(sz[0], sz[1], -1)
        hid = self.input_fc(x)

        for _layer in self.res_layers:
            hid = hid + F.leaky_relu(_layer(hid))

        return self.project(hid), x_fix
