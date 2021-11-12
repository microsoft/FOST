# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

name_list = ['mae', 'mse']

def mse(y_pred, y_true):
    """ Mean squared error regression loss.

    Parameters
    ----------
    y_pred: float
        predicted value
    y_true: float
        true value

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).
    """
    return ((y_pred-y_true)**2).mean()

def mae(y_pred, y_true):
    """ Mean absolute error regression loss.

    Parameters
    ----------
    y_pred: float
        predicted value
    y_true: float
        true value

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).
    """
    return (y_pred-y_true).abs().mean()

def build_loss(name):
    """ build loss function

    Parameters
    ----------
    name: String
        currently support 'mae' and 'mse'
        
    Returns
    -------
    func
        The function corresponds to the input name
    """
    assert name in name_list, f'only {name_list} loss support!'
    return eval(name)