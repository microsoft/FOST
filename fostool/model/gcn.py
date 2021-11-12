# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gconv import GATConv, EGNNConv


class BaseGNNNet(nn.Module):
    """Generic class for GNN Net.

    Inherited classes should implement dataflow_forward or subgraph_forward depended on
    different graph sampling strategies.

    """

    def __init__(self):
        super().__init__()

    def dataflow_forward(self, X, g):
        """
        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def subgraph_forward(self, X, g):
        """
        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def forward(self, X, g, **kwargs):
        """We support two types of graph sampling.
        dataflow: sampling recursively from top to bottom to formalize minibatches.
        subgraph: sampling only once and propogate through the subgraph to formalize minibatches.

        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Returns
        -------
        torch.Tensor
            forecasts

        Raises
        ------
        Exception
            Unsupported graph type
        """
        if g['type'] == 'dataflow':
            return self.dataflow_forward(X, g, **kwargs)
        elif g['type'] == 'subgraph':
            return self.subgraph_forward(X, g, **kwargs)
        else:
            raise Exception('Unsupported graph type {}'.format(g['type']))


class GATNet(BaseGNNNet):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='max', normalize='none'):
        """Build a GAT module

        Parameters
        ----------
        in_channels : int or tuple
            Size of each input sample
        out_channels : int
            Size of each output sample
        edge_channels : int
            Size of edge feature
        aggr : str, optional
            The aggregation scheme, by default 'max'
        normalize : str, optional
            The normalizing operator, by default 'none'
        """
        super().__init__()
        self.conv1 = GATConv(in_channels,
                             out_channels,
                             edge_channels,
                             aggr=aggr,
                             normalize=normalize)
        self.conv2 = GATConv(out_channels,
                             out_channels,
                             edge_channels,
                             aggr=aggr,
                             normalize=normalize)

    def dataflow_forward(self, X, g):
        """Forward logics for dataflow-based graph batches

        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Raises
        ------
        NotImplementedError
        """
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        size = g['size']
        res_n_id = g['res_n_id']

        c1 = self.conv1(
            (X, X[res_n_id[0]]), edge_index[0], edge_attr=edge_attr[0], size=size[0]
        )
        c1 = F.leaky_relu(c1)

        c2 = self.conv2(
            (c1, c1[res_n_id[1]]
             ), edge_index[1], edge_attr=edge_attr[1], size=size[1]
        )
        c2 = F.leaky_relu(c2)

        return c2

    def subgraph_forward(self, X, g):
        """Forward logics for subgraph-based graph batches

        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Raises
        ------
        NotImplementedError
        """
        res_n_id = g['res_n_id'].clone().detach()
        cent_n_id = g['cent_n_id'].clone().detach()
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        if 'edge_norm' in g:
            edge_norm = g['edge_norm']
        else:
            edge_norm = None
        c1 = self.conv1(X, edge_index, edge_attr=edge_attr,
                        edge_norm=edge_norm)
        c1 = F.leaky_relu(c1)
        c2 = self.conv2(c1, edge_index, edge_attr=edge_attr,
                        edge_norm=edge_norm)
        c2 = F.leaky_relu(c2)
        g['res_n_id'] = res_n_id
        g['cent_n_id'] = cent_n_id
        return c2


class EGNNNet(BaseGNNNet):
    def __init__(self, in_channels, out_channels, edge_channels, aggr='max', normalize='none'):
        """
        Build a EGNNNet module

        Parameters
        ----------
        in_channels : int or tuple
            Size of each input sample
        out_channels : int
            Size of each output sample
        edge_channels : int
            Size of edge feature
        aggr : str, optional
            The aggregation scheme, by default 'max'
        normalize : str, optional
            The normalizing operator, by default 'none'
        """

        super().__init__()
        self.conv1 = EGNNConv(in_channels, out_channels, edge_channels,
                              aggr=aggr, normalize=normalize)
        self.conv2 = EGNNConv(out_channels, out_channels, edge_channels,
                              aggr=aggr, normalize=normalize)

    def dataflow_forward(self, X, g):
        """
        Forward logics for dataflow-based graph batches

        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Raises
        ------
        NotImplementedError
        """
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        size = g['size']
        n_id = g['n_id']
        res_n_id = g['res_n_id']

        c1 = self.conv1((X, X[res_n_id[0]]), edge_index[0],
                        edge_attr=edge_attr[0],
                        size=size[0], indices=n_id[0][res_n_id[0]])
        c1 = F.leaky_relu(c1)

        c2 = self.conv2((c1, c1[res_n_id[1]]), edge_index[1],
                        edge_attr=edge_attr[1],
                        size=size[1], indices=n_id[1][res_n_id[1]])
        c2 = F.leaky_relu(c2)

        return c2

    def subgraph_forward(self, X, g):
        """Forward logics for subgraph-based graph batches

        Parameters
        ----------
        X : torch.Tensor
            input data
        g : dict
            graph info

        Raises
        ------
        NotImplementedError
        """
        edge_index = g['edge_index']
        edge_attr = g['edge_attr']
        n_id = g['cent_n_id']
        if 'edge_norm' in g:
            edge_norm = g['edge_norm']
        else:
            edge_norm = None

        c1 = self.conv1(X, edge_index,
                        edge_attr=edge_attr,
                        indices=n_id, edge_norm=edge_norm)
        c1 = F.leaky_relu(c1)

        c2 = self.conv2(c1, edge_index,
                        edge_attr=edge_attr,
                        indices=n_id, edge_norm=edge_norm)
        c2 = F.leaky_relu(c2)

        return c2
