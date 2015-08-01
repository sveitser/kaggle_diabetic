from __future__ import division, print_function
from collections import Counter
from datetime import datetime
from glob import glob
import os
import pprint

import click
import numpy as np
import pandas as pd
import theano
from sklearn.metrics import confusion_matrix, make_scorer

from sklearn.preprocessing import StandardScaler

from nolearn.lasagne import NeuralNet
from lasagne.updates import *

from ordinal_classifier import OrdinalClassifier
import data
import nn

from layers import *
from nn import *
from config import mkdir

np.random.seed(42)

#theano.sandbox.cuda.use("cpu")

START_MOM = 0.9
STOP_MOM = 0.95
START_LR = 0.0005
END_LR = START_LR * 0.001
L1 = 2e-5
L2 = 0.005
N_ITER = 100
PATIENCE = 20
POWER = 0.5
N_HIDDEN_1 = 32
N_HIDDEN_2 = 32
BATCH_SIZE = 128

SCHEDULE = {
    60: START_LR / 10.0,
    80: START_LR / 100.0,
    90: START_LR / 1000.0,
    N_ITER: 'stop'
}

RESAMPLE_WEIGHTS = [1.360, 14.37, 6.637, 40.23, 49.61]

class BlendNet(Net):
    def set_split(self, files, labels):
        """Override train/test split method."""
        def split(X, y, eval_size):
            if eval_size:
                tr, te = data.split_indices(files, labels, eval_size)
                return X[tr], X[te], y[tr], y[te]
            else:
                return X, y, X[len(X):], y[len(y):]
        setattr(self, 'train_test_split', split)


class ResampleIterator(BatchIterator):

    def __init__(self, batch_size, resample_prob=0.2, shuffle_prob=0.5):
        self.resample_prob = resample_prob
        self.shuffle_prob = shuffle_prob
        super(ResampleIterator, self).__init__(batch_size)

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        indices = data.balance_per_class_indices(self.y.ravel(), 
                                                 weights=RESAMPLE_WEIGHTS)
        for i in range((n_samples + bs - 1) // bs):
            r = np.random.rand()
            if r < self.resample_prob:
                sl = indices[np.random.randint(0, n_samples, size=bs)]
            elif r < self.shuffle_prob:
                sl = np.random.randint(0, n_samples, size=bs)
            else:
                sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)


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


class AdjustPower(object):
    def __init__(self, name, start=START_LR, power=POWER):
        self.name = name
        self.start = start
        self.power = power
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = self.start * np.array(
                [(1.0 - float(n) / nn.max_epochs) ** self.power
                for n in range(nn.max_epochs)])

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def get_nonlinearity():
    return np.random.choice([rectify, leaky_rectify, very_leaky_rectify,
                             sigmoid])

def rrange(a, b):
    return (np.random.rand() + a) * (b - a)

def get_estimator(n_features, files, labels, eval_size=0.1):
    layers = [
        (InputLayer, {'shape': (None, n_features)}),
        (DenseLayer, {'num_units': N_HIDDEN_1, 'nonlinearity': rectify,
                      'W': init.Orthogonal('relu'), 
                      'b':init.Constant(0.01)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': N_HIDDEN_2, 'nonlinearity': rectify,
                      'W': init.Orthogonal('relu'), 
                      'b':init.Constant(0.01)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': 1, 'nonlinearity': None}),
    ]
    args = dict(
        update=adam,
        update_learning_rate=theano.shared(float32(START_LR)),
        batch_iterator_train=ResampleIterator(BATCH_SIZE),
        batch_iterator_test=BatchIterator(BATCH_SIZE),
        objective=get_objective(l1=L1, l2=L2),
        eval_size=eval_size,
        custom_score=('kappa', util.kappa) if eval_size > 0.0 else None,
        on_epoch_finished = [
            Schedule('update_learning_rate', SCHEDULE),
        ],
        regression=True,
        max_epochs=N_ITER,
        verbose=1,
    )
    net = BlendNet(layers, **args)
    net.set_split(files, labels)
    return net

@click.command()
@click.option('--cnf', default='configs/c_128_4x4_32.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
@click.option('--per_patient', is_flag=True, default=False)
@click.option('--transform_file', default=None)
@click.option('--directory', default='data/features')
@click.option('--n_iter', default=1)
def fit(cnf, predict, per_patient, transform_file, n_iter, directory):

    config = util.load_module(cnf).config
    image_files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(image_files)
    labels = data.get_labels(names).astype(np.float32)[:, np.newaxis]

    tf_dirs = glob('{}/*/'.format(directory))
    tf_files = glob('{}/*.*'.format(directory))

    if transform_file is None:
        X_trains = [data.load_transform(directory)
                    for directory in tf_dirs] \
                   + [data.load_transform(transform_file=transform_file)
                      for transform_file in tf_files]
    else:
        X_trains = [data.load_transform(transform_file=transform_file)]

    scalers = [StandardScaler() for _ in X_trains]
    X_trains = [scaler.fit_transform(X_train) 
                for scaler, X_train in zip(scalers, X_trains)]

    if predict:

        if transform_file is None:
            X_tests = [data.load_transform(directory=directory, test=True)
                       for directory in tf_dirs]
        else:
            transform_file = transform_file.replace('train', 'test')
            X_tests = [data.load_transform(test=True, 
                                           transform_file=transform_file)]

        X_tests = [scaler.transform(X_test) 
                   for scaler, X_test in zip(scalers, X_tests)]

    # data.split_indices split per patient by default now
    tr, te = data.split_indices(image_files, labels)

    if not predict:
        print("feature matrix {}".format(X_train.shape))

        y_preds = []
        y_preds_train = []
        for i in range(n_iter):
            for X_train in X_trains:

                X = data.per_patient_reshape(X_train) \
                    if per_patient else X_train
                print('iter {}'.format(i))
                print('fitting split training set')
                est = get_estimator(X.shape[1], image_files, labels)
                est.fit(X, labels)

                y_pred = est.predict(X[te]).ravel()
                y_preds.append(y_pred)
                y_pred = np.mean(y_preds, axis=0)
                y_pred  = np.clip(np.round(y_pred).astype(int),
                                  np.min(labels), np.max(labels))

                print('kappa', i, util.kappa(labels[te], y_pred))
                print(confusion_matrix(labels[te], y_pred))

    if predict:

        y_preds = []
        y_preds_train = []
        for i in range(n_iter):
            for X_train, X_test in zip(X_trains, X_tests):
                print('fitting full training set')
                X = per_patient_reshape(X_train) if per_patient else X_train
                Xt = data.per_patient_reshape(X_test) \
                    if per_patient else X_test
                est = get_estimator(X.shape[1], eval_size=0.0, 
                                    randomize=RANDOMIZE)
                est.fit(X, labels)
                y_pred = est.predict(Xt).ravel()
                y_preds.append(y_pred)

                #y_pred_train = est.predict(X).ravel()
                #y_preds_train.append(y_pred_train)
                #tresh, kappa = get_best_thresholds(np.mean(y_preds_train,
                #                                           axis=0), 
                #                                   labels)
                #print('best kappa treshholds', tresh)
                #print('best kappa train', kappa)

        #tresh, kappa = get_best_thresholds(np.mean(y_preds_train, axis=0), 
        #                                   labels)
        #print('best kappa treshholds', tresh)
        #print('best kappa train', kappa)
        y_pred = np.mean(y_preds, axis=0)
        #y_pred = labels_from_thresholds(tresh, y_pred)
        y_pred  = np.clip(np.round(y_pred),
                          np.min(labels), np.max(labels)).astype(int)

        submission_filename = data.get_submission_filename()
        image_files = data.get_image_files(config.get('test_dir', TEST_DIR))
        names = data.get_names(image_files)
        image_column = pd.Series(names, name='image')
        level_column = pd.Series(y_pred, name='level')
        predictions = pd.concat([image_column, level_column], axis=1)

        print(predictions.tail())

        predictions.to_csv(submission_filename, index=False)
        print("saved predictions to {}".format(submission_filename))


if __name__ == '__main__':
    fit()
