# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from .input_fixer import InputFixer


class CNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        """Build a basic CNN encoder

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        kernel_size : int
            The size of convolutional kernels
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # set padding to ensure the same length
        # it is correct only when kernel_size is odd, dilation is 1, stride is 1
        self.conv = nn.Conv1d(
            input_dim, output_dim, kernel_size, padding=(kernel_size-1)//2
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            Updated representations
        """
        # input shape: [batch_size, node_num, seq_len, input_dim]
        # output shape: [batch_size, node_num, seq_len, input_dim]
        batch_size, node_num, seq_len, input_dim = x.shape
        x = x.view(-1, seq_len, input_dim).permute(0, 2, 1)
        y = self.conv(x)  # [batch_size*node_num, output_dim, conved_seq_len]
        y = y.permute(0, 2, 1).view(batch_size, node_num, -1, self.output_dim)

        return y


class KRNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, node_num, dup_num):
        """Build K parallel RNNs

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        node_num : int
            The number of nodes
        dup_num : int
            The number of parallel RNNs
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_num = node_num
        self.dup_num = dup_num

        self.rnn_modules = nn.ModuleList()
        for _ in range(dup_num):
            self.rnn_modules.append(
                nn.GRU(input_dim, output_dim)
            )
        self.attn = nn.Embedding(node_num, dup_num)

    def forward(self, x, n_id):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        n_id : torch.Tensor
            Node indices

        Returns
        -------
        torch.Tensor
            Updated representations
        """
        # input shape: [batch_size, node_num, seq_len, input_dim]
        # output shape: [batch_size, node_num, seq_len, output_dim]
        batch_size, node_num, seq_len, input_dim = x.shape
        # [seq_len, batch_size*node_num, input_dim]
        x = x.view(-1, seq_len, input_dim).permute(1, 0, 2)

        hids = []
        for rnn in self.rnn_modules:
            h, _ = rnn(x)  # [seq_len, batch_size*node_num, output_dim]
            hids.append(h)
        # [seq_len, batch_size*node_num, output_dim, num_dups]
        hids = torch.stack(hids, dim=-1)

        attn = torch.softmax(self.attn(n_id), dim=-1)  # [node_num, num_dups]

        hids = hids.view(
            seq_len, batch_size, node_num,
            self.output_dim, self.dup_num
        )
        hids = torch.einsum('ijklm,km->ijkl', hids, attn)
        hids = hids.permute(1, 2, 0, 3)

        return hids


class CNNKRNNEncoder(nn.Module):
    def __init__(self, cnn_input_dim, cnn_output_dim, cnn_kernel_size, rnn_output_dim, rnn_node_num, rnn_dup_num):
        """Build an encoder composed of CNN and KRNN

        Parameters
        ----------
        cnn_input_dim : int
            The input dimension of CNN
        cnn_output_dim : int
            The output dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_output_dim : int
            The output dimension of KRNN
        rnn_node_num : int
            The number of nodes for KRNN
        rnn_dup_num : int
            The number of parallel duplicates for KRNN
        """
        super().__init__()

        self.cnn_encoder = CNNEncoderBase(
            cnn_input_dim, cnn_output_dim, cnn_kernel_size
        )
        self.krnn_encoder = KRNNEncoderBase(
            cnn_output_dim, rnn_output_dim, rnn_node_num, rnn_dup_num
        )

    def forward(self, x, n_id):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input data
        n_id : torch.Tensor
            Node indices

        Returns
        -------
        torch.Tensor
            Updated representations
        """
        cnn_out = self.cnn_encoder(x)
        krnn_out = self.krnn_encoder(cnn_out, n_id)

        return krnn_out


class KRNNModel(nn.Module):
    def __init__(self, fea_dim, cnn_dim, cnn_kernel_size, rnn_dim, num_nodes, rnn_dups, lookahead, lookback, **params):
        """Build a KRNN model

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
            The number of parallel duplicates
        lookahead : int
            The number of lookahead steps
        lookback : int
            The number of lookback steps
        """
        super().__init__()

        self.input_fixer = InputFixer(
            num_nodes=num_nodes,
            fea_dim=fea_dim
        )

        fea_dim = 1

        self.encoder = CNNKRNNEncoder(
            cnn_input_dim=fea_dim,
            cnn_output_dim=cnn_dim,
            cnn_kernel_size=cnn_kernel_size,
            rnn_output_dim=rnn_dim,
            rnn_node_num=num_nodes,
            rnn_dup_num=rnn_dups
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

        encode = self.encoder(x, g['cent_n_id'])
        encode, _ = encode.max(dim=2)
        out = self.out_fc(encode)

        return out, x_fix
