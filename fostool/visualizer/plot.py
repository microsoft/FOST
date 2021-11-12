# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
# from matplotlib import pyplot as plt
from matplotlib.dates import (
    AutoDateLocator,
    AutoDateFormatter,
)
from matplotlib.ticker import FuncFormatter

# overwrites matplotlib's built-in datetime plotting with pandas datetime plotting
pd.plotting.register_matplotlib_converters()

# disable log messages from "matplotlib"
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger('Plot')
logger.setLevel(logging.INFO)


def plot(
    train_data, predict_data, lookahead, xlabel='Date', ylabel='Y',
    figsize=(10, 6), include_legend=True, lookback_size=None, node_name=None
):
    """plot predicted data by splicing the training data of specific length(lookback_size) ahead.

    Parameters 
    ---------- 
    train_data : dataframe 
        the dataframe of train data.
    predict_data : dataframe
        Predicted results by function "predict" of class "Pipeline"
    lookahead : int
        The length of your predicted data
    xlabel : str, optional
        xlabel is used to set the label for the x-axis
    ylabel : str, optional
        ylabel is used to set the label for the y-axis
    figsize : Tuple(float, float), optional
        figsize of figure, (width, height) in inches.
    include_legend : Boolean, optional
        Whether or not to place a legend on the Axes.
    lookback_size : int, optional
        The length of training data that you want to splice with the predicted data(count from back to front)
    node_name : str, optional
        name of Node that you want to plot(default is the name of first Node)

    Returns 
    ----------  
    Figure
        The figure you set up
    """

    if not lookback_size or not isinstance(lookback_size, int) or lookback_size < 0:
        # generate default lookback_size if None
        magnification = 5
        lookback_size = lookahead * magnification
        logger.info(
            'Unspecified lookback_size, use default lookback_size: {}.'.format(lookback_size))

    predict_data = predict_data.reset_index()
    nodename_set = predict_data['Node'].unique()
    # generate default node_name if None
    if not node_name:
        node_name = predict_data['Node'].loc[0]
    elif node_name not in nodename_set:
        raise Exception("Node_name not in Node")

    fig = plt.figure(facecolor='w', figsize=figsize)
    ax = fig.add_subplot(111)

    predict_data = predict_data[predict_data["Node"] == node_name]

    if not isinstance(train_data, pd.DataFrame) or train_data.empty:
        logger.warning('train_data is empty')
    else:
        selected_train_data = train_data[train_data["Node"].isin([node_name])]
        selected_train_data = selected_train_data[-lookback_size:]

        ax.plot(selected_train_data['Date'],
                selected_train_data['TARGET'], ls='-',  label='Train')

        predict_data = pd.concat([selected_train_data[-1:], predict_data],
                                 axis=0, ignore_index=True)

    ax.plot(predict_data['Date'],
            predict_data['TARGET'], ls='-',  label='Predict')

    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if include_legend:
        ax.legend()   
    fig.tight_layout()

    plt.title('{} data prediction'.format(node_name))
    plt.show()

    return fig
