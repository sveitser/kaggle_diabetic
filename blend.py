from __future__ import division, print_function
from collections import Counter
from datetime import datetime
import os
import pprint

import click
import numpy as np
import pandas as pd
import theano
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

#theano.sandbox.cuda.use("cpu")

#MIN_LEARNING_RATE = 0.000001
#MAX_MOMENTUM = 0.9721356783919598
START_MOM = 0.9
STOP_MOM = 0.99
#INIT_LEARNING_RATE = 0.00002
START_LR = 0.0002
END_LR = START_LR * 0.001
ALPHA = 0.002
N_ITER = 200
PATIENCE = 20
POWER = 0.5

def epsilon_insensitive(y, t, d0=0.05, d1=0.5):
    #return T.maximum(epsilon**2.0, (y - t)**2.0) - epsilon ** 2.0
    #return T.maximum((abs(y - t) - epsilon)**2.0, 0.0)
    #return T.maximum(abs(y/eps - t/eps), (y/eps - t/eps)**2) * eps
    #return T.switch(T.lt(abs(y - t), eps), abs(y - t), (y - t)**2 / eps)
    #return T.switch(T.lt(abs(y - t), eps), 0.5 * (y - t)**2.0, 
    #                                       eps * (abs(y - t) - 0.5 * eps))
    a = abs(y - t)
    #huber = T.switch(T.lt(a, d1), (a - d0)**2.0, a - d1 + (d1 - d0)**2.0)
    #return T.switch(T.lt(a, d0), 0.0, huber)
    return T.switch(T.lt(a, d0), 0.0, (a - d0)**2.0)

def shuffle(*arrays):
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]

class ResampleIterator(BatchIterator):
    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            #sl = slice(i * bs, (i + 1) * bs)
            sl = np.random.choice(np.arange(0, n_samples), bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

class ShufflingBatchIteratorMixin(object):
    def __iter__(self):
        if not hasattr(self, 'count'):
            self.count = 0
            self.interval = 1
        self.count += 1
        if self.count % self.interval == 0:
            print('shuffle')
            self.interval = self.count * 2
            self.X, self.y = shuffle(self.X, self.y)
        for X, y in super(ShufflingBatchIteratorMixin, self).__iter__():
            #X = X + np.random.randn(*X.shape).astype(np.float32) * 0.05
            yield X, y

class ShuffleIterator(ShufflingBatchIteratorMixin, BatchIterator):
    pass

class RegularizedObjective(Objective):
    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):

        loss = super(RegularizedObjective, self).get_loss(
            input=input, target=target, deterministic=deterministic, **kwargs)
        if not deterministic:
            return loss \
                + ALPHA * lasagne.regularization.l2(self.input_layer)
        else:
            return loss

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


def get_estimator(n_features, **kwargs):
    l = [
        (InputLayer, {'shape': (None, n_features)}),
        #(DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 32, 'nonlinearity': very_leaky_rectify,
                      'W': init.Orthogonal('relu'), 'b':init.Constant(0.1)}),
        #(DropoutLayer, {'p': 0.5}),
        (DenseLayer, {'num_units': 32, 'nonlinearity': leaky_rectify,
                      'W': init.Orthogonal('relu'), 'b':init.Constant(0.1)}),
        #(DropoutLayer, {'p': 0.5}),
        #(DenseLayer, {'num_units': 128, 'nonlinearity': leaky_rectify}),
        (DenseLayer, {'num_units': 1, 'nonlinearity': None}),
    ]
    args = dict(
    
        update=nesterov_momentum,
        #update=rmsprop,
        #update=adadelta,

        update_learning_rate=theano.shared(float32(START_LR)),
        update_momentum=theano.shared(float32(START_MOM)),

        #batch_iterator_train=ShuffleIterator(128),
        batch_iterator_train=ResampleIterator(128),

        objective=RegularizedObjective,
        #objective_loss_function=epsilon_insensitive,

        eval_size=0.1,
        custom_score=('kappa', util.kappa) \
            if kwargs.get('eval_size', 0.1) > 0.0 else None,

        on_epoch_finished = [
            AdjustPower('update_learning_rate', start=START_LR),
            AdjustVariable('update_momentum', start=START_MOM, stop=STOP_MOM),
            #AdjustPower('update_momentum', start=START_MOM, power=0.5),
            #AdjustLearningRate('update_learning_rate', loss='kappa', 
            #                   greater_is_better=True, patience=PATIENCE,
            #                   save=False),
        ],

        regression=True,
        max_epochs=N_ITER,
        verbose=1,
    )
    args.update(kwargs)
    return Net(l, **args)

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

    scaler = StandardScaler()

    if per_patient:
        X_train = per_patient_reshape(X_train)
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        

    if predict:

        if transform_file is not None:
            transform_file = transform_file.replace('train', 'test')
        X_test = load_transform(test=True, transform_file=transform_file)

        if per_patient:
            X_test = per_patient_reshape(X_test)

        X_test = scaler.transform(X_test).astype(np.float32)

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
            est = get_estimator(X_train.shape[1], eval_size=0.0)
            est.fit(X_train, labels)
            y_pred = est.predict(X_test).ravel()
            y_preds.append(y_pred)

        y_pred = np.mean(y_preds, axis=0)
        y_pred  = np.clip(np.round(y_pred),
                          np.min(labels), np.max(labels)).astype(int)

        submission_filename = util.get_submission_filename()
        files = util.get_image_files(model.get('test_dir', TEST_DIR))
        names = util.get_names(files)
        image_column = pd.Series(names, name='image')
        level_column = pd.Series(y_pred, name='level')
        predictions = pd.concat([image_column, level_column], axis=1)

        print(predictions.tail())

        predictions.to_csv(submission_filename, index=False)
        print("saved predictions to {}".format(submission_filename))


if __name__ == '__main__':
    fit()
