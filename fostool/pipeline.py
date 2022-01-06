# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import torch

from .model import TSModel
from .dataset.data_utils import DataHandler
#from tools.infer import Predictor
from .tools.trainer import Trainer
from .tools.predictor import Predictor
from .visualizer.plot import plot
from .task.config_handler import YamlHandler
from .task.fusion import Fusion
from .task.loss import build_loss

from datetime import date, datetime
from .task.logger import setLogger


setLogger()
logger = logging.getLogger('Pipeline')
logger.setLevel(logging.INFO)


############################################################################################################################


class Pipeline(object):
    """ST forecaster.
    """

    def __init__(self, lookahead, train_path=None, config_path=None, graph_path=None, load_path=None, lookback_list='auto', **params):
        """
        Parameters
        ----------
        lookahead : int
            Forward steps you would like to predict.
        train_path : str or DataFrame, optional
            Training path or DataFrame
        config_path : str, optional
            Model and strategy config path
        graph_data : str, default None, optional
            Graph data path or dataframe, None if not exist
        load_path : str, optional
            Load existing Pipeline form path
        lookback_list : 'auto' or list[int], optional
            Backward steps used in model training, suggest remains 'auto' if you are not formiliar with this.
        """
        if not config_path:
            parent = os.path.dirname(__file__)
            config_path = os.path.join(parent, 'config/default.yaml')
        config = YamlHandler(config_path).read_yaml()

        self._init_base_config(lookahead=lookahead,
                               lookback_list=lookback_list, **config.base)
        self.node_data = None
        if load_path is None:
            assert train_path is not None, "train path is required for model trainning."
            self._init_data(train_path, graph_path, **config.data_handler)
            run_id = 0
            if config.base.dump_path[-1] == '/':
                config.base.dump_path = config.base.dump_path[:-1]
            os.makedirs(config.base.dump_path, exist_ok=True)
            folders = ['_'.join(x.split('_')[:-6])
                       for x in os.listdir(config.base.dump_path)]

            while config.base.dump_path+f'/{run_id}' in folders:
                run_id += 1
            timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            run_id = f'{run_id}_{timestamp}'
            config.base.dump_path += f'/{run_id}'
            os.makedirs(os.path.join(config.base.dump_path,
                                     'weights'), exist_ok=True)
        self._init_trainer(config.base.dump_path, **config.trainer, **params)
        self._init_model(config.model)
        self._init_fusion(**config.fusion)
        self.config = config

        self.graph_path = graph_path
        if load_path:
            self.load_pipeline(load_path)

    def _init_trainer(self, dump_path, **params):
        self.trainer = Trainer(dump_path, **params)

    def _split_data(self, data, lookahead, lookback):
        """split data to train set and valid set
        """
        train_data = data.groupby(level=0, group_keys=False).apply(
            lambda x: x.iloc[:-lookahead]
        )

        valid_data = data.groupby(level=0, group_keys=False).apply(
            lambda x: x.iloc[-(lookahead + lookback):]
        )

        return train_data, valid_data

    def _init_data(self, train_path, graph_path, **params):
        """Load and preprocess data
        """
        self.train_path = train_path
        assert isinstance(train_path, pd.DataFrame) or isinstance(
            train_path, str), 'train_path should be str or dataframe'

        if isinstance(train_path, pd.DataFrame):
            train_path['Date'] = pd.to_datetime(train_path['Date'])
            node_data = train_path
        elif isinstance(train_path, str):
            node_data = pd.read_csv(train_path, parse_dates=['Date'])

        if graph_path is not None:
            if isinstance(graph_path, pd.DataFrame):
                graph_data = graph_path
            elif isinstance(train_path, str):
                graph_data = pd.read_csv(graph_path)
        else:
            unique_node = node_data['Node'].unique()

            if len(unique_node) < 100:
                graph_data = pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [unique_node, unique_node], names=['node_0', 'node_1']
                    )
                ).reset_index()
            else:
                graph_data = pd.DataFrame(
                    {'node_0': unique_node, 'node_1': unique_node}
                )
            graph_data['weight'] = 1.0

        if 'Unnamed: 0' in node_data.columns:
            node_data = node_data.drop('Unnamed: 0', axis=1)
        if 'Unnamed: 0' in graph_data.columns:
            graph_data = graph_data.drop('Unnamed: 0', axis=1)

        self.node_data, self.graph_data, self.meta_info =\
            DataHandler.preprocess(
                node_data, graph_data
            )

    def _init_model(self, model_configs):
        """build config for different sub models
        """
        self.model_config_dict = {}
        self.model_dict = {}

        self.lookback_list.sort()

        for lookback_size in self.lookback_list:
            for model_class_name, model_config in model_configs.items():
                model_config = copy.deepcopy(model_config)
                model_config.update(
                    {"lookback": lookback_size, "lookahead": self.lookahead}
                )
                self.model_config_dict[(model_class_name, lookback_size)] \
                    = model_config

    def _init_base_config(self, lookahead=7, target='TARGET', freq='auto', lookback_list='auto', seed=42, loss_func='mse', **params):
        """initialize configs
        Parameters
        ----------
        lookahead : int, optional
            Forward steps you would like to predict.
        target : str, optional
            Target column name
        freq : str or int, optional
            Sampling frequency, suggest remains 'auto' if you are not formiliar with this.
        lookback_list : str or list[int], optional
            Backward steps used in model training
        seed : int, optional
            Random seed, not used in current version
        loss_func : str, optional
            loss function, support 'mae' or 'mse' in current version
        """
        self.lookahead = lookahead
        self.target = target
        self.freq = freq

        if lookback_list == 'auto':
            if self.lookahead < 7:
                self.lookback_list = [7, 14, 21]
            else:
                self.lookback_list = [
                    self.lookahead * (l + 1) for l in range(3)]
        else:
            self.lookback_list = lookback_list

        self.loss_func = build_loss(loss_func)

    def _init_fusion(self, method='average', **params):
        """ 
        Parameters
        ----------
        method : str, optional
            Ensemble methods, only support 'average' in this version
        """
        self.fusion = Fusion(method)

    def fit(self, **params):
        """ST fit
        """
        for (model_class_name, lookback),  model_params in self.model_config_dict.items():
            train_node_data, valid_node_data = self._split_data(
                self.node_data,
                lookahead=self.lookahead,
                lookback=lookback
            )

            train_data_handler = DataHandler(
                train_node_data,
                self.graph_data,
                lookahead=self.lookahead,
                mode='train',
            )

            num_samples = train_data_handler.get_num_samples(lookback)

            if num_samples < 100:
                print('Not enough training data. Will skip the model training.')
                break

            train_data_loader, model_meta_info = train_data_handler.build_data_loader(
                lookback=lookback
            )

            valid_data_handler = DataHandler(
                valid_node_data,
                self.graph_data,
                lookahead=self.lookahead,
                mode='valid',
            )

            valid_data_loader = valid_data_handler.build_data_loader(
                lookback=lookback
            )

            self.meta_info['model_meta_info'][lookback] = model_meta_info
            model_params.update(model_meta_info)
            model = TSModel(model_class_name, model_params)

            exp_name = f'{model.name()}_{lookback}'
            model, val_loss = self.trainer(
                exp_name, train_data_loader, valid_data_loader, model, self.loss_func, **params
            )

            self.meta_info['val_loss_info'][exp_name] = val_loss
            self.model_dict[exp_name] = model

    def predict(self, test_path=None, **params):
        """Predict
        Parameters
        ----------
        test_path : str, optional
            If given, model will predict the given test data, otherwise train data will be used for predict
        """
        def _recover(df, time_anchor, meta_info):
            node_le = meta_info['node_le']
            mean = meta_info['mean'][[self.target]]
            std = meta_info['std'][[self.target]]
            sample_freq = meta_info['sample_freq']

            df['Node'] = node_le.inverse_transform(df['Node'])
            df['Date'] = df['Date'].map(
                lambda x: sample_freq * (x + 1) + time_anchor
            )

            df = df.set_index(['Node', 'Date'])
            df = df * std + mean

            return df.sort_index()

        if test_path is None:
            assert not self.node_data is None, 'For inference, you have to offer a test csv file'
            node_data = self.node_data
        else:
            node_data = pd.read_csv(test_path, parse_dates=['Date'])
            if 'Unnamed: 0' in node_data.columns:
                node_data = node_data.drop('Unnamed: 0', axis=1)
            node_data = DataHandler.node_transform(
                node_data, meta_info=self.meta_info
            )
            self.train_path = test_path

        predictor = Predictor()
        result = dict()
        for exp_name, model in self.model_dict.items():
            lookback = int(exp_name.split('_')[-1])
            _node_data = DataHandler.make_predict_data(
                node_data,
                lookback=lookback,
                lookahead=self.lookahead,
                sample_freq=self.meta_info['sample_freq']
            )

            test_data_handler = DataHandler(
                _node_data,
                self.graph_data,
                lookahead=self.lookahead,
                mode='test',
            )
            test_data_loader = test_data_handler.build_data_loader(lookback)
            time_anchor = node_data.index.get_level_values(1).max()

            _result_df = predictor(test_data_loader, model, self.target)
            _result_df = _recover(_result_df, time_anchor, self.meta_info)

            result[exp_name] = _result_df

        result = self.fusion(result, loss_info=self.meta_info['val_loss_info'])
        return result

    def fit_and_predict(self, dump_weight=True, **params):
        """Fit and predict
        """
        self.fit(dump_weight)
        return self.predict()

    def save_pipeline(self, path=None):
        """Save pipeline

        Parameters
        ----------
        path : str, optional
            Path to save Pipeline

        """
        if path is None:
            path = self.config.base.dump_path
        pd.to_pickle(self.meta_info, f'{path}/meta_info.pkl')
        for exp_name, model in self.model_dict.items():
            model.dump_weight(f'{path}/weights/{exp_name}.pth')
        return path

    def load_pipeline(self, path=None):
        """Load pipeline

        Parameters
        ----------
        path : str, optional
            Path to load Pipeline

        """
        self.model_dict = {}
        if path is None:
            path = self.config.base.dump_path
        self.meta_info = pd.read_pickle(f'{path}/meta_info.pkl')
        for (model_class_name, lookback), model_params in self.model_config_dict.items():
            model_params.update(self.meta_info['model_meta_info'][lookback])
            model = TSModel(model_class_name, model_params)
            exp_name = f'{model.name()}_{lookback}'
            model.load_weight_from_path(f'{path}/weights/{exp_name}.pth')
            self.model_dict[exp_name] = model
        self.graph_data = self.meta_info['graph_data']

    def plot(self, predict_data, node_name=None, lookback_size=None, train_data=None, **params):
        """plot predicted data by splicing the training data of specific length(lookback_size) ahead.

        Parameters 
        ---------- 
        predict_data : dataframe
            Predicted results by function "predict" of class "Pipeline"
        node_name : str
            name of Node that you want to plot(default is the name of first Node)
        lookahead : int
            The length of your predicted data
        lookback_size : int
            The length of training data that you want to splice with the predicted data(count from back to front)
        train_data : dataframe 
            the dataframe of train data, if None, data in self.train_path will be used as train_data.
        xlabel : str
            xlabel is used to set the label for the x-axis
        ylabel : str
            ylabel is used to set the label for the y-axis
        figsize : Tuple(float, float)
            figsize of figure, (width, height) in inches.
        include_legend : Boolean
            Whether or not to place a legend on the Axes.

        Returns 
        ----------  
        Figure
            The figure you set up
        """
        if not isinstance(train_data, pd.DataFrame) and not (self.train_path is None):
            if isinstance(self.train_path, pd.DataFrame):
                train_data = self.train_path[['Node', 'Date', 'TARGET']]
            else:
                train_data = pd.read_csv(self.train_path, usecols=['Node', 'Date', 'TARGET'], parse_dates=[
                    'Date'], index_col=False)[['Node', 'Date', 'TARGET']]
        plot(train_data, predict_data, self.lookahead,
             node_name=node_name, lookback_size=lookback_size, **params)
