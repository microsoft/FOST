# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GATNet, EGNNNet
from .krnn import CNNKRNNEncoder
from .input_fixer import InputFixer


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, gcn_type, aggr, normalize):
        """Build a basic GCN block

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
        """
        super(GCNBlock, self).__init__()
        GCNClass = {
            'gat': GATNet,
            'egnn': EGNNNet,
        }.get(gcn_type)
        self.gcn = GCNClass(
            in_channels,
            out_channels,
            edge_channels,
            aggr=aggr,
            normalize=normalize
        )

    def forward(self, X, g):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        g : dict
            Graph sample

        Returns
        -------
        torch.Tensor
            Updated representations
        """

        batch_size, node_num, seq_len, fea_num = X.shape
        t1 = X.permute(1, 0, 2, 3).contiguous().\
            view(node_num, batch_size*seq_len, fea_num)
        t2 = F.relu(self.gcn(t1, g))
        out = t2.view(node_num, batch_size, seq_len, -1).\
            permute(1, 0, 2, 3).contiguous()
        return out


class SandwichEncoder(nn.Module):
    def __init__(self,
                 fea_dim,
                 cnn_dim,
                 cnn_kernel_size,
                 rnn_dim,
                 num_nodes,
                 rnn_dups,
                 gcn_dim,
                 edge_fea_dim,
                 gcn_type,
                 gcn_aggr,
                 gcn_norm,
                 ):
        """Build a sandwich encoder

        Parameters
        ----------
        fea_dim : int
            The feature dimension
        cnn_dim : int
            The hidden dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_dim : int
            The hidden dimension of KRNN
        num_nodes : int
            The number of nodes
        rnn_dups : int
            The number of parallel RNN duplicates
        gcn_dim : int
            The hidden dimension of GCN
        edge_fea_dim : int
            The dimension of edge features
        gcn_type : str
            The type of GCN
        gcn_aggr : str
            The type of GCN aggregation
        gcn_norm : str
            The type of GCN normalization
        """
        super().__init__()

        self.first_encoder = CNNKRNNEncoder(cnn_input_dim=fea_dim,
                                            cnn_output_dim=cnn_dim,
                                            cnn_kernel_size=cnn_kernel_size,
                                            rnn_output_dim=rnn_dim,
                                            rnn_node_num=num_nodes,
                                            rnn_dup_num=rnn_dups
                                            )
        self.gcn = GCNBlock(in_channels=rnn_dim,
                            out_channels=gcn_dim,
                            edge_channels=edge_fea_dim,
                            gcn_type=gcn_type,
                            aggr=gcn_aggr,
                            normalize=gcn_norm)
        self.second_encoder = CNNKRNNEncoder(cnn_input_dim=gcn_dim,
                                             cnn_output_dim=cnn_dim,
                                             cnn_kernel_size=cnn_kernel_size,
                                             rnn_output_dim=rnn_dim,
                                             rnn_node_num=num_nodes,
                                             rnn_dup_num=rnn_dups)

    def forward(self, x, g):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        g : dict
            Graph sample

        Returns
        -------
        torch.Tensor
            Updated representations
        """
        if g['type'] == 'dataflow':
            first_out = self.first_encoder(x, g['graph_n_id'])
        elif g['type'] == 'subgraph':
            first_out = self.first_encoder(x, g['cent_n_id'])
        else:
            raise Exception('Unsupported graph type: {}'.format(g['type']))
        gcn_out = self.gcn(first_out, g)
        second_out = self.second_encoder(gcn_out, g['cent_n_id'])
        encode_out = first_out + second_out
        return encode_out


class SandwichModel(nn.Module):
    def __init__(self,
                 fea_dim,
                 cnn_dim,
                 cnn_kernel_size,
                 rnn_dim,
                 num_nodes,
                 rnn_dups,
                 gcn_dim,
                 edge_fea_dim,
                 gcn_type,
                 gcn_aggr,
                 gcn_norm,
                 lookahead,
                 **params
                 ):
        """Build a Sandwich model

        Parameters
        ----------
        fea_dim : int
            The feature dimension
        cnn_dim : int
            The hidden dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_dim : int
            The hidden dimension of KRNN
        num_nodes : int
            The number of nodes
        rnn_dups : int
            The number of parallel RNN duplicates
        gcn_dim : int
            The hidden dimension of GCN
        edge_fea_dim : int
            The dimension of edge features
        gcn_type : str
            The type of GCN
        gcn_aggr : str
            The type of GCN aggregation
        gcn_norm : str
            The type of GCN normalization
        lookahead : int
            The number of lookahead steps
        """
        super().__init__()

        self.input_fixer = InputFixer(
            num_nodes=num_nodes,
            fea_dim=fea_dim
        )

        fea_dim = 1

        self.encoder = SandwichEncoder(
            fea_dim, cnn_dim, cnn_kernel_size, rnn_dim, num_nodes, rnn_dups, gcn_dim, edge_fea_dim, gcn_type, gcn_aggr, gcn_norm
        )

        self.out_fc = nn.Linear(rnn_dim, lookahead)

    def forward(self, x, g):
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

        encode = self.encoder(x, g)
        pool, _ = encode.max(dim=2)

        out = self.out_fc(pool)
        return out, x_fix
