# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from collections import OrderedDict


class GATConv(PyG.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='max', normalize='none', **kwargs):
        """Build a gated graph convolutional layer.

        Parameters
        ----------
        in_channels : (int or tuple)
            Size of each input sample
        out_channels : int
            Size of each output sample
        edge_channels : (int)
            Size of edge feature
        aggr : str, optional
            The aggregation operator, by default 'max'
        normalize : str, optional
            The normalizing operator, by default 'none'
        **kwargs : optional
            Additional arguments for PyG.MessagePassing
        """
        super().__init__(aggr=aggr, node_dim=-3, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.u = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.v = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.normalize = normalize

        if normalize == 'bn':
            self.batch_norm = nn.BatchNorm1d(out_channels)
        if normalize == 'ln':
            self.layer_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x, edge_index, edge_attr=None, edge_norm=None, size=None):
        """
        Parameters
        ----------
        x : torch.Tensor or (torch.Tensor, torch.Tensor)
            Input data
        edge_index : torch.Tensor
            The index of edges
        edge_attr : torch.Tensor or None, optional
            Edge attributes, by default None
        edge_norm : str or None, optional
            The normalization type for edges, by default None
        size : int, optional
            The output dimension, by default None

        Returns
        -------
        torch.Tensor
            The enriched representations after message passing
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = [None if xi is None else torch.matmul(
                xi, self.weight_n) for xi in x]
        else:
            x = torch.matmul(x, self.weight_n)

        edge_attr = torch.matmul(edge_attr, self.weight_e)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr, edge_norm=edge_norm)

    def message(self, edge_index_i, x_i, x_j, edge_attr, edge_norm):
        """
        Parameters
        ----------
        edge_index_i : torch.Tensor
            The index of target nodes
        x_i : torch.Tensor
            The representations of target nodes indexed by edge_index_i
        x_j : torch.Tensor
            The representations of source nodes indexed by edge_index_j
        edge_attr : torch.Tensor
            Edge attributes
        edge_norm : torch.Tensor
            The normalization for edges

        Returns
        -------
        torch.Tensor
            Messages in edges
        """
        x_i = torch.matmul(x_i, self.u)
        x_j = torch.matmul(x_j, self.u)
        gate = torch.sigmoid((x_i * x_j).sum(dim=-1)).unsqueeze(dim=-1)
        msg = x_j * gate
        if edge_norm is None:
            return msg
        else:
            return msg * edge_norm.reshape(edge_norm.size(0), 1, 1)

    def update(self, aggr_out, x):
        """
        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated messages
        x : torch.Tensor
            Raw inputs

        Returns
        -------
        torch.Tensor
            Updated representations

        Raises
        ------
        KeyError
            Unsupported normalization type
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[1]

        if self.normalize == 'bn':
            aggr_out = aggr_out.permute(0, 2, 1)
            aggr_out = self.batch_norm(aggr_out)
            aggr_out = aggr_out.permute(0, 2, 1)
        elif self.normalize == 'ln':
            aggr_out = self.layer_norm(aggr_out)
        elif self.normalize == 'none':
            aggr_out = aggr_out
        else:
            raise KeyError(
                (f'not support normalize type: {self.normalize}')
            )

        return x + aggr_out


class EGNNConv(PyG.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,  aggr='max', normalize='none', **kwargs):
        """Build an edge-attribute-aware graph convolutional layer.

        Parameters
        ----------
        in_channels : (int or tuple)
            Size of each input sample
        out_channels : int
            Size of each output sample
        edge_channels : (int)
            Size of edge feature
        aggr : str, optional
            The aggregation operator, by default 'max'
        normalize : str, optional
            The normalizing operator, by default 'none'
        **kwargs : optional
            Additional arguments for PyG.MessagePassing
        """
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.query = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.key = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.linear_att = nn.Linear(3 * out_channels, 1)
        self.linear_out = nn.Linear(2 * out_channels, out_channels)

        self.normalize = normalize

        if normalize == 'bn':
            self.batch_norm = nn.BatchNorm1d(out_channels)
        if normalize == 'ln':
            self.layer_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key)

    def forward(self, x, edge_index, edge_attr, size=None, indices=None, edge_norm=None):
        """
        Parameters
        ----------
        x : torch.Tensor or (torch.Tensor, torch.Tensor)
            Input data
        edge_index : torch.Tensor
            The index of edges
        edge_attr : torch.Tensor or None, optional
            Edge attributes, by default None
        size : int, optional
            The output dimension, by default None
        indicies: torch.Tensor or None, optional
            The node indices
        edge_norm : str or None, optional
            The normalization type for edges, by default None

        Returns
        -------
        torch.Tensor
            The enriched representations after message passing
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = [None if xi is None else torch.matmul(
                xi, self.weight_n) for xi in x]
        else:
            x = torch.matmul(x, self.weight_n)

        edge_attr = torch.matmul(edge_attr, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr, indices=indices, edge_norm=edge_norm)

    def message(self, x_j, x_i, edge_attr, edge_norm):
        """
        Parameters
        ----------
        x_j : torch.Tensor
            The representations of source nodes
        x_i : torch.Tensor
            The representations of target nodes
        edge_attr : torch.Tensor
            Edge attributes
        edge_norm : torch.Tensor
            The normalization factors for edges

        Returns
        -------
        torch.Tensor
            The messages in edges
        """
        # cal att of shape [B, E, 1]
        query = torch.matmul(x_j, self.query)
        key = torch.matmul(x_i, self.key)

        edge_attr = edge_attr.unsqueeze(dim=1).expand_as(query)

        att_feature = torch.cat([query, key, edge_attr], dim=-1)
        att = torch.sigmoid(self.linear_att(att_feature))

        # gate of shape [1, E, C]
        gate = torch.sigmoid(edge_attr)

        msg = att * x_j * gate

        if edge_norm is None:
            return msg
        else:
            return msg * edge_norm.reshape(edge_norm.size(0), 1, 1)

    def update(self, aggr_out, x, indices):
        """
        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated messages
        x : torch.Tensor
            Raw inputs
        indices: torch.Tensor
            Node indexes

        Returns
        -------
        torch.Tensor
            Updated representations

        Raises
        ------
        KeyError
            Unsupported normalization type
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[1]

        aggr_out = self.linear_out(torch.cat([x, aggr_out], dim=-1))

        if self.normalize == 'bn':
            aggr_out = aggr_out.permute(0, 2, 1)
            aggr_out = self.batch_norm(aggr_out)
            aggr_out = aggr_out.permute(0, 2, 1)
        elif self.normalize == 'ln':
            aggr_out = self.layer_norm(aggr_out)
        elif self.normalize == 'none':
            aggr_out = aggr_out
        else:
            raise KeyError(
                (f'not support normalize type: {self.normalize}')
            )

        return x + aggr_out
