from __future__ import print_function
from collections import Counter
import cPickle as pickle
from datetime import datetime
import pprint
from time import time
import sys

import pandas as pd
import numpy as np

import lasagne
import lasagne.layers
from lasagne import init

from lasagne.updates import *
from lasagne import updates
from lasagne.objectives import (MaskedObjective, Objective,
                                categorical_crossentropy, mse)
from lasagne.layers import get_all_layers, get_output, InputLayer
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import SaveWeights
import theano
from theano import tensor as T

from quadratic_weighted_kappa import quadratic_weighted_kappa
import data
import util
import iterator


def float32(k):
    return np.cast['float32'](k)


def create_net(config, **kwargs):
    args = {
        'layers': config.layers,
        'batch_iterator_train': iterator.ResampleIterator(
            config, batch_size=config.get('batch_size_train')),
        'batch_iterator_test': iterator.SharedIterator(
            config, batch_size=config.get('batch_size_test')),
        'on_epoch_finished': [
            Schedule('update_learning_rate', config.get('schedule')),
            SaveBestWeights(loss='kappa', greater_is_better=True,),
            SaveWeights(config.weights_epoch, every_n_epochs=5),
            SaveWeights(config.weights_best, every_n_epochs=1, only_best=True),
        ],
        'objective': get_objective(),
        'use_label_encoder': False,
        'eval_size': 0.1,
        'regression': True,
        'max_epochs': 1000,
        'verbose': 2,
        'update_learning_rate': theano.shared(
            float32(config.get('schedule')[0])),
        'update': nesterov_momentum,
        'update_momentum': 0.9,
        'custom_score': ('kappa', util.kappa),

    }
    args.update(kwargs)
    net = Net(**args)
    net._config = config
    return net


def get_objective(l1=0.0, l2=0.0005):
    class RegularizedObjective(Objective):

        def get_loss(self, input=None, target=None, aggregation=None,
                     deterministic=False, **kwargs):

            l1_layer = get_all_layers(self.input_layer)[1]

            loss = super(RegularizedObjective, self).get_loss(
                input=input, target=target, aggregation=aggregation,
                deterministic=deterministic, **kwargs)
            if not deterministic:
                return loss \
                    + l1 * lasagne.regularization.regularize_layer_params(
                        l1_layer, lasagne.regularization.l1) \
                    + l2 * lasagne.regularization.regularize_network_params(
                        self.input_layer, lasagne.regularization.l2)
            else:
                return loss
    return RegularizedObjective


class Schedule(object):
    def __init__(self, name, schedule):
        self.name = name
        self.schedule = schedule

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        if epoch in self.schedule:
            new_value = self.schedule[epoch]
            if new_value == 'stop':
                if hasattr(nn, '_config'):
                    nn.save_params_to(nn._config.final_weights_file)
                raise StopIteration
            getattr(nn, self.name).set_value(float32(new_value))


class SaveBestWeights(object):
    def __init__(self, loss='kappa', greater_is_better=True):
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.loss = loss
        self.greater_is_better = greater_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.loss] \
            * (-1.0 if self.greater_is_better else 1.0)
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
            nn.save_params_to(nn._config.weights_file)


class Net(NeuralNet):

    def train_test_split(self, X, y, eval_size):
        if eval_size:
            X_train, X_valid, y_train, y_valid = data.split(
                X, y, test_size=eval_size)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = X[len(X):], y[len(y):]

        return X_train, X_valid, y_train, y_valid

    def initialize(self):
        if getattr(self, '_initialized', False):
            return

        out = getattr(self, '_output_layer', None)
        if out is None:
            out = self._output_layer = self.initialize_layers()
        self._check_for_unused_kwargs()

        iter_funcs = self._create_iter_funcs(
            self.layers_, self.objective, self.update,
            self.y_tensor_type,
            )
        self.train_iter_, self.eval_iter_, self.predict_iter_, self.transform_iter_ = iter_funcs
        self._initialized = True

    def _create_iter_funcs(self, layers, objective, update, output_type):
        y_batch = output_type('y_batch')

        output_layer = list(layers.values())[-1]
        objective_params = self._get_params_for('objective')
        obj = objective(output_layer, **objective_params)
        if not hasattr(obj, 'layers'):
            # XXX breaking the Lasagne interface a little:
            obj.layers = layers

        loss_train = obj.get_loss(None, y_batch)
        loss_eval = obj.get_loss(None, y_batch, deterministic=True)
        predict_proba = get_output(output_layer, None, deterministic=True)

        try:
            transform = get_output([v for k, v in layers.items() 
                                   if 'rmspool' in k or 'maxpool' in k][-1],
                                   None, deterministic=True)
        except IndexError:
            transform = get_output(layers.values()[-2], None,
                                   deterministic=True)

        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = self.get_all_params(trainable=True)
        update_params = self._get_params_for('update')
        updates = update(loss_train, all_params, **update_params)

        input_layers = [layer for layer in layers.values()
                        if isinstance(layer, InputLayer)]

        X_inputs = [theano.Param(input_layer.input_var, name=input_layer.name)
                    for input_layer in input_layers]
        inputs = X_inputs + [theano.Param(y_batch, name="y")]

        train_iter = theano.function(
            inputs=inputs,
            outputs=[loss_train],
            updates=updates,
            )
        eval_iter = theano.function(
            inputs=inputs,
            outputs=[loss_eval, accuracy],
            )
        predict_iter = theano.function(
            inputs=X_inputs,
            outputs=predict_proba,
            )
        transform_iter = theano.function(
            inputs=X_inputs,
            outputs=transform,
            )
        return train_iter, eval_iter, predict_iter, transform_iter

    def transform(self, X, transform=None, color_vec=None):
        features = []
        for Xb, yb in self.batch_iterator_test(X, transform=transform,
                                               color_vec=color_vec):
            features.append(self.transform_iter_(Xb))
        return np.vstack(features)
    
    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        class_losses = None

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            y_pred, y_true = [], []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                if not np.isfinite(batch_train_loss[0]):
                    raise ValueError("non finite loss")
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):

                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)

                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)
                y_true.append(yb)
                if self.custom_score:
                    y_prob = self.predict_iter_(Xb)
                    y_pred.append(y_prob)

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if self.custom_score and self.eval_size:

                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                y_pred = np.clip(y_pred, np.min(y_true), np.max(y_true))
                avg_custom_score = self.custom_score[1](y_true, y_pred)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
            if self.custom_score and self.eval_size:
                info[self.custom_score[0]] = avg_custom_score
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

