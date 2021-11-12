# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import os
import torch
import numpy as np
from .utils import decorate_batch
import logging
logger = logging.getLogger('Trainer')
logger.setLevel(logging.INFO)


class Trainer(object):
    def __init__(self, dump_path, device='cuda', max_epochs=100):
        """
        Parameters
        ----------
        dump_path : str
            path to dump middle model weight
        device: str, optional
            'cpu' or 'cuda'
        max_epochs: int, optional
            max training epoch
        """
        super().__init__()

        if device == 'cuda':
            assert torch.cuda.is_available(), 'No cuda device found.'

        self.dump_path = dump_path
        self.device = device
        self.max_epochs = max_epochs
        self.early_stop_patients = 10

    def _set_trainer_state(self, exp_name):
        self._exp_dir = os.path.join(self.dump_path, 'tmp', exp_name)
        self._model_path = os.path.join(self._exp_dir, 'model.cpt')
        os.makedirs(self._exp_dir, exist_ok=True)

        self._best_val_loss = float('inf')  # the best validation score so far
        self._best_val_epoch = -1  # the epoch for the best validation loss

    def _judge_best_epoch(self, val_loss, epoch):
        is_best_epoch = False
        is_early_stop = False
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_val_epoch = epoch
            is_best_epoch = True
        elif epoch - self._best_val_epoch >= self.early_stop_patients:
            is_early_stop = True

        return is_best_epoch, is_early_stop

    def _dump_checkpoint(self, model, is_best_epoch):
        if is_best_epoch:
            model.dump_weight(self._model_path)

    def _load_checkpoint(self, model):
        model.load_weight_from_path(self._model_path)

    def build_optimizer(self, torch_model):
        # TODO: add learning rate and optimizer config
        return torch.optim.AdamW(torch_model.parameters(), lr=1e-3, weight_decay=1e-5)

    def __call__(self, exp_name, train_data_loader, valid_data_loader, model, loss_func):
        """
        Parameters
        ----------
        exp_name : str
            experiment name
        train_data_loader: pytorch DataLoader
            dataLoader for training data
        valid_data_loader: pytorch DataLoader
            dataLoader for validation data
        model: TSModel
            model for training
        loss_func: func
            loss function used in training
        
        Return
        ----------
        model: pytorch Model
            trained model
        
        _best_val_loss: float
            best validation score
        """
        # preparations before training
        self._set_trainer_state(exp_name)
        # build optimizer
        torch_model = model.model
        torch_model.to(self.device)

        optimizer = self.build_optimizer(torch_model)

        for epoch in range(self.max_epochs):
            # training
            torch_model.train()

            train_losses = []
            val_losses = []

            for batch_idx, batch in enumerate(train_data_loader):
                batch = decorate_batch(batch, self.device)
                (x, y), g, _ = batch

                y_hat, _ = torch_model(x, g)
                loss = loss_func(y_hat, y)
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                torch_model.zero_grad()

            train_loss = np.mean(train_losses)

            # validation
            torch_model.eval()
            for batch_idx, batch in enumerate(valid_data_loader):
                batch = decorate_batch(batch, self.device)
                (x, y), g, _ = batch
                with torch.no_grad():
                    y_hat, _ = torch_model(x, g)
                loss = loss_func(y_hat, y)
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)

            logger.info('On epoch {}, train loss {}, val loss {}'.format(
                epoch, train_loss, val_loss))
            logger.info('-' * 12)

            is_best_epoch, is_early_stop = self._judge_best_epoch(
                val_loss, epoch
            )
            if is_best_epoch:
                logger.info('For model {}, current best val loss {}'
                            .format(exp_name, val_loss)
                            )

            self._dump_checkpoint(model, is_best_epoch)
            if is_early_stop:
                break

        self._load_checkpoint(model)

        return model, self._best_val_loss
