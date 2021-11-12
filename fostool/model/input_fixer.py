# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

node_start_index = 0

time_start_index = 1
time_end_index = 6


class InputFixer(nn.Module):
    def __init__(self, num_nodes, fea_dim):
        """Build an input fixer

        Parameters
        ----------
        num_nodes : int
            The number of nodes
        fea_dim : int
            The dimension of node features
        """
        super(InputFixer, self).__init__()

        self.num_nodes = num_nodes
        self.fea_dim = (fea_dim - 6) // 2

        self.embed_dim = 3

        self._build()

    def _build(self):
        """Build internal modules
        """
        self._build_embed()

        self.d = nn.Linear(self.embed_dim * 6 + self.fea_dim, self.fea_dim)
        self.c = nn.Linear(self.embed_dim * 6 + self.fea_dim, self.fea_dim)

        self.s = nn.Parameter(torch.Tensor(self.fea_dim))

        self.proj = nn.Linear(self.fea_dim, 1)

        nn.init.constant_(self.c.bias, 3)
        nn.init.constant_(self.s, 1e-1)

    def _build_embed(self):
        """Build internal embedding tables
        """
        self.node_em = nn.Embedding(self.num_nodes, self.embed_dim)

        self.month_em = nn.Embedding(12, self.embed_dim)
        self.day_em = nn.Embedding(31, self.embed_dim)
        self.weekday_em = nn.Embedding(7, self.embed_dim)
        self.hour_em = nn.Embedding(24, self.embed_dim)
        self.minute_em = nn.Embedding(60, self.embed_dim)

    def embed_time(self, time):
        """Transform time info into embeddings

        Parameters
        ----------
        time : torch.Tensor
            Time info

        Returns
        -------
        torch.Tensor
            time embedding
        """
        month = self.month_em(time.select(-1, 0) - 1)
        day = self.day_em(time.select(-1, 1) - 1)
        weekday = self.weekday_em(time.select(-1, 2))
        hour = self.hour_em(time.select(-1, 3))
        minute = self.minute_em(time.select(-1, 4))

        return torch.cat([month, day, weekday, hour, minute], dim=-1)

    def embed_node(self, node):
        """Transform node ids into embeddings

        Parameters
        ----------
        node : torch.Tensor
            Node indicies

        Returns
        -------
        torch.Tensor
            Node embeddings
        """
        return self.node_em(node)

    def _get_embed(self, node, time):
        """Extract embeddings for a specific node at a specific timestamp

        Parameters
        ----------
        node : torch.Tensor
            The node index
        time : torch.Tensor
            The timestamp

        Returns
        -------
        (torch.Tensor, torch.Tensor)
            A tuple of node and time representations
        """
        node = self.embed_node(node)
        time = self.embed_time(time)
        return node, time

    def forward(self, x):
        """The workflow of the input fixer

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            Updated input data
        """
        node = x[:, :, :, node_start_index].long()
        time = x[:, :, :, time_start_index: time_end_index].long()

        feat_gap = x[:, :, :, time_end_index:].float()
        feat, gap = torch.split(feat_gap, self.fea_dim, dim=-1)

        node, time = self._get_embed(node, time)

        g = torch.exp(-gap * torch.pow(self.s, 2))

        c = torch.sigmoid(
            self.c(torch.cat([node, time, feat], dim=-1))
        )

        # reliable score
        c = c * g

        d = self.d(
            torch.cat([node, time, feat], dim=-1)
        )

        feat = feat * c + d * (1 - c)

        return feat.select(-1, -1).unsqueeze(dim=-1)
