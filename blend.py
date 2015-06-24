from __future__ import division, print_function
from collections import Counter
from datetime import datetime
import os
import pprint

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.grid_search import GridSearchCV
import xgboost as xgb


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nolearn.lasagne import NeuralNet
from lasagne.updates import *

from ordinal_classifier import OrdinalClassifier
import util
import nn

from definitions import *
from boost import *
from layers import *
from nn import *


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def get_estimator(n_features):
    l = [
        (InputLayer, {'shape': (None, n_features)}),
        #(DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 128, 'nonlinearity': leaky_rectify}),
        (DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 128, 'nonlinearity': leaky_rectify}),
        #(DropoutLayer, {'p': 0.5}),
        #(DenseLayer, {'num_units': 128, 'nonlinearity': leaky_rectify}),
        (DenseLayer, {'num_units': 1, 'nonlinearity': None}),
    ]

    return Net(
        l,
        update=nesterov_momentum,
        #update=rmsprop,

        update_learning_rate=theano.shared(float32(0.01)),
        update_momentum=theano.shared(float32(0.9)),

        objective=RegularizedObjective,

        eval_size=0.1,
        custom_score=('kappa', util.kappa),

        on_epoch_finished = [
            AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
            AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],

        regression=True,
        max_epochs=80,
        verbose=1,
    )

@click.command()
@click.option('--cnf', default='config/large.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
@click.option('--grid_search', is_flag=True, default=False)
@click.option('--per_patient', is_flag=True, default=False)
@click.option('--transform_file', default=None)
@click.option('--n_iter', default=1)
def fit(cnf, predict, grid_search, per_patient, transform_file, n_iter):

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names).astype(np.float32)[:, np.newaxis]

    X_train = load_transform(transform_file=transform_file)


    if per_patient:
        X_train = per_patient_reshape(X_train).astype(np.float32)

    if predict:

        if transform_file is not None:
            transform_file = transform_file.replace('train', 'test')
        X_test = load_transform(test=True, transform_file=transform_file)

        if per_patient:
            X_test = per_patient_reshape(X_test).astype(np.float32)

    # util.split_indices split per patient by default now
    tr, te = util.split_indices(labels)

    # 
    if not predict:
        print("feature matrix {}".format(X_train.shape))

        if grid_search:
            kappa_scorer = make_scorer(util.kappa)
            gs = GridSearchCV(est, grid, verbose=3, cv=[(tr, te)], 
                              scoring=kappa_scorer, n_jobs=1, refit=False)
            gs.fit(X_train, labels)
            pd.set_option('display.height', 500)
            pd.set_option('display.max_rows', 500)
            df = util.grid_search_score_dataframe(gs)
            print(df)
            df.to_csv('grid_scores.csv')
            df.to_csv('grid_scores_{}.csv'.format(datetime.now().isoformat()))
            #est = gs.best_estimator_
        else:
            y_preds = []
            for i in range(n_iter):
                print('iter {}'.format(i))
                print('fitting split training set')
                est = get_estimator(X_train.shape[1])
                est.fit(X_train, labels)
                y_pred = est.predict(X_train[te]).ravel()
                y_preds.append(y_pred)
             
                y_pred = np.mean(y_preds, axis=0)
                y_pred  = np.clip(np.round(y_pred).astype(int),
                                  np.min(labels), np.max(labels))
                print(labels[te].ravel().shape, y_pred.shape)
                print('kappa', i, util.kappa(labels[te], y_pred))
                print(confusion_matrix(labels[te], y_pred))

    if predict:

        y_preds = []
        for i in range(n_iter):
            print('fitting full training set')
            est = get_estimator(X_train.shape[1])
            est.fit(X_train, labels)
            y_pred = est.predict(X_test).ravel()
            y_preds.append(y_pred)

        y_pred = np.mean(y_preds, axis=0)
        y_pred  = np.clip(np.round(y_pred).astype(int),
                          np.min(labels), np.max(labels))

        submission_filename = util.get_submission_filename()
        files = util.get_image_files(model.get('test_dir', TEST_DIR))
        names = util.get_names(files)
        image_column = pd.Series(names, name='image')
        level_column = pd.Series(y_pred, name='level')
        predictions = pd.concat([image_column, level_column], axis=1)
        predictions.to_csv(submission_filename, index=False)
        print("saved predictions to {}".format(submission_filename))

if __name__ == '__main__':
    fit()
