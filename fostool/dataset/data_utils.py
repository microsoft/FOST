# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import os
import json
import copy
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset

from torch_sparse import SparseTensor, rw, saint
from torch_geometric.data import Data, ClusterData, ClusterLoader, NeighborSampler

from sklearn.preprocessing import LabelEncoder
import logging
logger = logging.getLogger('DataUtils')
logger.setLevel(logging.INFO)


class RandomWalkSampler(object):
    def __init__(self, data, batch_size, walk_length=2):
        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                value=data.edge_attr, sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None
        self.data.edge_attr = None

        self.batch_size = batch_size
        self.walk_length = walk_length

    def __sample_nodes__(self):
        """Sampling initial nodes by iterating over the random permutation of nodes"""
        tmp_map = torch.arange(self.N, dtype=torch.long)
        all_n_id = torch.randperm(self.N, dtype=torch.long)
        node_samples = []
        for s_id in range(0, self.N, self.batch_size):
            init_n_id = all_n_id[s_id:s_id+self.batch_size]  # [batch_size]

            n_id = self.adj.random_walk(init_n_id, self.walk_length)
            n_id = n_id.flatten().unique()  # [num_nodes_in_subgraph]
            tmp_map[n_id] = torch.arange(n_id.size(0), dtype=torch.long)
            res_n_id = tmp_map[init_n_id]

            node_samples.append((n_id, res_n_id))

        return node_samples

    def __sample__(self, num_epoches):
        samples = []
        for _ in range(num_epoches):
            node_samples = self.__sample_nodes__()
            for n_id, res_n_id in node_samples:
                adj, e_id = self.adj.saint_subgraph(n_id)
                samples.append((n_id, e_id, adj, res_n_id))

        return samples

    def __get_data_from_sample__(self, sample):
        n_id, e_id, adj, res_n_id = sample
        data = self.data.__class__()
        data.num_nodes = n_id.size(0)
        row, col, value = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        data.n_id = n_id
        data.res_n_id = res_n_id
        data.e_id = e_id

        return data

    def __len__(self):
        return (self.N + self.batch_size-1) // self.batch_size

    def __iter__(self):
        for sample in self.__sample__(1):
            data = self.__get_data_from_sample__(sample)
            yield data


class GraphDataset(IterableDataset):
    def __init__(self, tensors, edge_index, edge_attr, batch_size=16, saint_batch_size=100, saint_walk_length=2, shuffle=False):
        self.tensors = tensors

        self.num_nodes = self.tensors[0].size(1)

        assert(self.num_nodes == edge_index.max(
        ) + 1), f'num nodes {self.num_nodes} != edge_index max {edge_index.max()}'

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.batch_size = batch_size
        self.saint_batch_size = saint_batch_size
        self.saint_walk_length = saint_walk_length
        self.shuffle = shuffle

        self._graph_sampler = self._make_graph_sampler()
        self._length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_attr, num_nodes=self.num_nodes
        )

        sampler = RandomWalkSampler(
            graph, self.saint_batch_size, walk_length=self.saint_walk_length
        )

        return sampler

    def get_subgraph(self, subgraph):
        graph = {
            'type': 'subgraph',
            'edge_index': subgraph.edge_index,
            'edge_attr': subgraph.edge_attr,
            'cent_n_id': subgraph.n_id,
            'res_n_id': subgraph.res_n_id,
            'e_id': subgraph.e_id,
        }

        return graph

    def __iter__(self):
        for subgraph in self._graph_sampler:
            g = self.get_subgraph(subgraph)

            tensors = list(
                map(lambda x: x[:, g['cent_n_id']], self.tensors)
            )

            dataset_len = tensors[0].size(0)
            indices = list(range(dataset_len))

            if self.shuffle:
                np.random.shuffle(indices)

            num_batches = (len(indices) + self.batch_size - 1) \
                // self.batch_size

            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                yield list(map(lambda x: x[indices[start: end]], tensors)), g, torch.LongTensor(indices[start: end])

    def get_length(self):
        num_samples_per_node = self.tensors[0].size(0)

        length = (num_samples_per_node + self.batch_size - 1) \
            // self.batch_size
        length *= len(self._graph_sampler)

        return length

    def __len__(self):
        return self._length


class DataHandler:
    def __init__(self, node_data, graph_data, lookahead, trunc_ratio=5, mode='train', batch_size=32):
        """Data handler class

        Parameters
        ----------
        node_data : DataFrame
            time series data
        graph_data : DataFrame
            static graph
        lookahead : int
            forecast horizon
        trunc_ratio : int, optional
            truncate historical time series within trunc_ratio * lookahead, by default 5
        mode : str, optional
            'train', 'valid' or 'test', by default 'train'
        batch_size : int, optional
            batch size of data loader, by default 32
        """
        assert mode in ['train', 'test', 'valid']

        if mode == 'train':
            self.shuffle = True
        else:
            self.shuffle = False

        self.mode = mode
        self.lookahead = lookahead

        self.trunc_ratio = trunc_ratio
        self.batch_size = batch_size

        self.node_data = copy.deepcopy(node_data)
        self.graph_data = copy.deepcopy(graph_data)

    @classmethod
    def node_transform(cls, node_data, meta_info):
        """map node names to index and perform z-score to time series
        """        
        # re-order the columns
        columns = node_data.columns.tolist()
        columns.remove('TARGET')
        columns.append('TARGET')

        node_data = node_data[columns]

        node_le = meta_info['node_le']

        mean = meta_info['mean'].reset_index()
        std = meta_info['std'].reset_index()

        mean['Node'] = node_le.transform(mean['Node'])
        std['Node'] = node_le.transform(std['Node'])

        mean = mean.set_index('Node')
        std = std.set_index('Node')

        sample_freq = meta_info['sample_freq']

        node_data['Node'] = node_le.transform(node_data['Node'])
        node_data = node_data.set_index(['Node', 'Date'])

        node_data = cls._standard_normalize(node_data, mean, std)

        node_data = _fill_missing_time(node_data, sample_freq)
        node_data = _fill_missing_value(node_data)

        return node_data

    @classmethod
    def graph_transform(cls, graph_data, meta_info):
        """map node name to index
        """        
        node_le = meta_info['node_le']

        graph_data['node_0'] = node_le.transform(
            graph_data['node_0']
        )
        graph_data['node_1'] = node_le.transform(
            graph_data['node_1']
        )

        return graph_data

    @classmethod
    def preprocess(cls, node_data, graph_data):
        # return processed data and meta_info
        meta_info = dict()

        # node label encoder
        node_le = LabelEncoder()
        node_le.fit(
            graph_data[['node_0', 'node_1']].values.reshape(-1, )
        )

        mean = node_data.groupby('Node').mean()
        std = node_data.groupby('Node').std()

        sample_freq = _infer_sample_freq(node_data)

        meta_info.update(
            {
                'node_le': node_le,
                'mean': mean,
                'std': std,
                'sample_freq': sample_freq,
                'model_meta_info': dict(),
                'val_loss_info': dict()
            }
        )
        node_data = cls.node_transform(node_data, meta_info)
        graph_data = cls.graph_transform(graph_data, meta_info)

        meta_info.update(
            {
                'graph_data': graph_data
            }
        )

        return node_data, graph_data, meta_info

    @classmethod
    def _standard_normalize(cls, data, mean, std):
        for col in data.columns:
            if col in mean:
                data[col] = (data[col] - mean[col]) / (std[col] + 1e-3)

        # clip to 5 times standard deviation
        return data.clip(-5, 5)

    @classmethod
    def make_predict_data(cls, data, lookback, lookahead, sample_freq):
        """generate data for the testing phase
        """        
        data = data.sort_index()
        data = data.groupby(level=0, group_keys=False).apply(
            lambda x: x.iloc[-lookback:]
        )

        start_time = data.index.get_level_values(-1).min()

        end_time = data.index.get_level_values(-1).max()
        end_time = end_time + sample_freq * lookahead

        data = _reindex_full_time(
            data, sample_freq, start_time, end_time
        )

        return data

    def get_num_samples(self, lookback):
        N = len(self.node_data.index.levels[0])
        T = len(self.node_data.index.levels[1])
        return N * (T - self.lookahead - lookback - 1)

    def build_data_loader(self, lookback):
        """return a pytorch data loader with specific lookback
        """        
        inputs, outputs = _build_input_tensor(
            self.node_data,
            self.graph_data,
            lookahead=self.lookahead,
            lookback=lookback,
            trunc_ratio=self.trunc_ratio,
            use_aug=(self.mode == 'train')
        )
        edge_weight_cols = [
            x for x in self.graph_data.columns if x.startswith('weight')
        ]
        edge_index = torch.LongTensor(
            self.graph_data[['node_0', 'node_1']].values.T
        )

        edge_attr = torch.FloatTensor(self.graph_data[edge_weight_cols].values)

        graph_dataset = GraphDataset(
            tensors=[inputs, outputs],
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

        if self.mode == 'train':
            model_meta_info = {}
            model_meta_info['fea_dim'] = inputs.shape[-1]
            model_meta_info['edge_fea_dim'] = len(edge_weight_cols)
            model_meta_info['num_nodes'] = self.node_data.index.get_level_values(
                'Node').unique().shape[0]

            return DataLoader(
                graph_dataset,
                batch_size=None
            ), model_meta_info
        else:
            return DataLoader(
                graph_dataset,
                batch_size=None
            )


def _infer_sample_freq(data):
    """infer the sampling frequency of node data
    """    
    def _infer_time_gap(x):
        def _check_dateoff(time_values, offset):
            prev_time = time_values[:-1].reset_index()['Date']
            after_time = time_values[1:].reset_index()['Date']
            intersect = np.sum(after_time == prev_time + offset)
            ratio = intersect / len(after_time)
            return ratio > 0.9

        x = x.sort_values(by='Date')
        time_values = x['Date']

        year_off = pd.tseries.offsets.DateOffset(years=1)
        month_off = pd.tseries.offsets.DateOffset(months=1)
        quarter_off = pd.tseries.offsets.DateOffset(months=3)

        for offset in [year_off, month_off, quarter_off]:
            if _check_dateoff(time_values, offset):
                return offset

        offset = time_values.sort_values().diff().min()

        if pd.isnull(offset):
            return None
        else:
            return pd.tseries.frequencies.to_offset(offset)

    time_gap = data.groupby('Node').apply(_infer_time_gap)
    time_gap = pd.DataFrame(time_gap).rename(columns={0: 'freq'})
    time_gap = time_gap.groupby('freq').size()
    time_gap = time_gap[time_gap == time_gap.max()].index[0]

    logger.info('Detected Sample Frequency: {}.'.format(time_gap))

    return time_gap


def _reindex_full_time(x, freq, start_time=None, end_time=None):
    """fullfill the missing time slots
    """
    _index = [x.index.get_level_values(0).unique()]

    if start_time is None:
        start_time = x.index.get_level_values(1).min()
        end_time = x.index.get_level_values(1).max()

    time_full = pd.date_range(start_time, end_time, freq=freq, name='Date')

    _index.append(time_full)

    _index = pd.MultiIndex.from_product(_index)

    return x.reindex(_index)


def _fill_missing_time(data, time_gap=None):
    logger.info('{} Rows Before Time Reindex.'.format(data.shape[0]))
    data = _reindex_full_time(data, time_gap)
    logger.info('{} Rows After Time Reindex.'.format(data.shape[0]))
    logger.info('-' * 20)

    return data


def _fill_missing_value(data):
    def _fill_missing(x):
        steps = np.asarray(
            [np.arange(x.shape[0]) for _ in range(x.shape[1])]
        ).T

        # time step of each sample
        ts = pd.DataFrame(steps, columns=x.columns, index=x.index)

        ts_na = pd.DataFrame(steps, columns=x.columns, index=x.index)
        ts_na[pd.isnull(x)] = np.nan

        # the time gap since the last observation
        gap_t = ts - ts_na.fillna(method='ffill').fillna(100)
        gap_t.columns = ['_gap_{}'.format(x) for x in gap_t.columns]

        # since we already normalize the data, we simply fill 0.0 for the missing rows in the BEGINNING
        x = x.fillna(method='ffill').fillna(0.0)

        x = pd.concat([x, gap_t], axis=1)

        return x

    logger.info('{} Rows Before Fill Missing.'.format(data.shape[0]))

    data = data.groupby(level=0, group_keys=False).apply(_fill_missing)

    logger.info('{} Rows After Fill Missing.'.format(data.shape[0]))
    logger.info('-' * 20)

    return data


def _parse_time(x):
    """parse time feature from given DataFrame
    """    
    ts = x.index.get_level_values(-1)

    month = ts.map(lambda x: x.month).values.reshape(-1, 1)
    day = ts.map(lambda x: x.day).values.reshape(-1, 1)
    weekday = ts.map(lambda x: x.weekday()).values.reshape(-1, 1)
    hour = ts.map(lambda x: x.hour).values.reshape(-1, 1)
    minute = ts.map(lambda x: x.minute).values.reshape(-1, 1)

    return np.concatenate([month, day, weekday, hour, minute], axis=-1)


def _build_input_tensor(
    node_data,
    graph_data,
    lookahead,
    lookback,
    trunc_ratio,
    use_aug=False
):
    """

    Parameters
    ----------
    use_aug : bool, optional
        if use_aug is True, we use the reverse augment for training, by default False

    Returns
    -------
    Tensor,Tensor
        inputs, outputs Tensor
    """
    node_data = node_data.sort_index()

    node_data = node_data.groupby(level=0, group_keys=False).apply(
        lambda x: x.iloc[-(lookback + lookahead) * trunc_ratio:]
    )

    # number of nodes
    N = len(node_data.index.levels[0])
    T = len(node_data.index.levels[1])
    F = len(node_data.columns)

    node = node_data.index.get_level_values(0)
    node = node.values.reshape(N, T, -1)
    node = torch.from_numpy(node)

    time = _parse_time(node_data)
    time = time.reshape(N, T, -1)
    time = torch.from_numpy(time)

    inputs, outputs = [], []

    if use_aug:
        # use reverse augmentation during training
        aug_func_list = [_origin, _reverse]
    else:
        aug_func_list = [_origin]

    for aug_func in aug_func_list:
        aug_data = node_data.copy()

        target = aug_data['TARGET'].values
        target = target.reshape(N, T, -1)
        target = aug_func(target)

        aug_data['TARGET'] = target.reshape(-1, )

        target = torch.from_numpy(target).squeeze(dim=-1)

        feat = aug_data.values
        feat = feat.reshape(N, T, -1)
        feat = torch.from_numpy(feat)

        input_tensor = torch.cat([node, time, feat], dim=-1)
        output_tensor = target

        for t in range(lookback, T - lookahead + 1):
            inputs.append(
                input_tensor[:, t - lookback: t, :].unsqueeze(dim=0)
            )
            outputs.append(
                output_tensor[:, t: t + lookahead].unsqueeze(dim=0)
            )

    inputs = torch.cat(inputs, dim=0).float()
    outputs = torch.cat(outputs, dim=0).float()

    return inputs, outputs


def _origin(x):
    return x


def _reverse(x):
    return -x
