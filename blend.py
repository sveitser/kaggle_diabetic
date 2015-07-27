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
from sklearn.grid_search import GridSearchCV
import xgboost as xgb


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from nolearn.lasagne import NeuralNet
from lasagne.updates import *
from lasagne.layers import get_all_layers

from ordinal_classifier import OrdinalClassifier
import iterator, nn, util

from definitions import *
from boost import *
from layers import *
from nn import *
from model import mkdir

np.random.seed(42)

#theano.sandbox.cuda.use("cpu")

BLEND_FEATURE_DIR = 'data/blend'
mkdir(BLEND_FEATURE_DIR)

#MIN_LEARNING_RATE = 0.000001
#MAX_MOMENTUM = 0.9721356783919598
START_MOM = 0.9
STOP_MOM = 0.95
#INIT_LEARNING_RATE = 0.00002
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
RANDOMIZE = False
INIT_W = init.Orthogonal('relu')
INIT_B = init.Constant(0.1)

SCHEDULE = {
    60: START_LR / 10.0,
    80: START_LR / 100.0,
    90: START_LR / 1000.0,
    N_ITER: 'stop'
}

TEST_SCHEDULE = {
    20: START_LR / 100.0,
    40: START_LR / 1000.0,
    50: 'stop',
}

RESAMPLE_WEIGHTS = np.array([1.360, 14.37, 6.637, 40.23, 49.61])
#RESAMPLE_WEIGHTS = np.array([1.360, 14.37, 6.637, 40.23, 100])
#FINAL_WEIGHTS = np.array([1, 2, 2, 2, 2])
#RESAMPLE_WEIGHTS = [1, 8, 5, 20, 100]
#RESAMPLE_PROB = 0.3
RESAMPLE_PROB = 0.2
SHUFFLE_PROB = 0.5
REGRESSION = True


def get_objective(l1=L1, l2=L2):
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

def epsilon_insensitive(y, t, d0=0.01, d1=0.5):
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

class TestIterator(BatchIterator):
    # make this work with the transform method in nn.py
    def __call__(self, X, y=None, **kwargs):
        return super(TestIterator, self).__call__(X, y)

class ResampleIterator(BatchIterator):

    def __iter__(self):
        #self.n_iter = getattr(self, 'n_iter', 0) + 1
        n_samples = self.X.shape[0]
        bs = self.batch_size
        #alpha = (0.5 - self.n_iter * 0.005) ** 4.0
        #w = alpha * RESAMPLE_WEIGHTS + (1 - alpha) * FINAL_WEIGHTS
        #indices = util.balance_per_class_indices(self.y.ravel(), weights=w)
        indices = util.balance_per_class_indices(self.y.ravel(), 
                                                 weights=RESAMPLE_WEIGHTS)

        for i in range((n_samples + bs - 1) // bs):
            r = np.random.rand()
            if r < RESAMPLE_PROB:
                sl = indices[np.random.randint(0, n_samples, size=bs)]
            elif r < SHUFFLE_PROB:
                sl = np.random.randint(0, n_samples, size=bs)
            else:
                sl = slice(i * bs, (i + 1) * bs)
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

def get_estimator(n_features, eval_size=0.1, randomize=False, test=False):
    n_hidden_1 = np.random.randint(12, 33) * 2 if randomize else N_HIDDEN_1
    n_hidden_2 = np.random.randint(12, 33) * 2 if randomize else N_HIDDEN_2
    nl1 = get_nonlinearity() if randomize else rectify
    nl2 = get_nonlinearity() if randomize else rectify
    l1 = np.random.lognormal(np.log(L1), 1) if randomize else L1
    l2 = np.random.lognormal(np.log(L2), 1) if randomize else L2
    l = [
        (InputLayer, {'shape': (None, n_features)}),
        #(DropoutLayer, {}),
        (DenseLayer, {'num_units': n_hidden_1, 'nonlinearity': nl1,
                      'W': init.Orthogonal('relu'), 
                      'b': init.Constant(0.1)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': n_hidden_2, 'nonlinearity': nl2,
                      'W': init.Orthogonal('relu'), 
                      'b': init.Constant(0.1)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        #(DenseLayer, {'num_units': 5, 'nonlinearity': softmax}),
        (DenseLayer, {'num_units': 1}),
    ]
    args = dict(
        #update=nesterov_momentum,
        update=adam,
        update_learning_rate=theano.shared(
            float32(START_LR / 10.0 if test else START_LR)),
        #update_momentum=theano.shared(float32(START_MOM)),

        batch_iterator_train=ResampleIterator(BATCH_SIZE),
        batch_iterator_test=TestIterator(BATCH_SIZE),

        objective=get_objective(l1=l1, l2=l2),

        eval_size=eval_size,
        custom_score=('kappa', util.kappa) if eval_size > 0.0 else None,
        on_epoch_finished = [
            Schedule('update_learning_rate', 
                     TEST_SCHEDULE if test else SCHEDULE),
        ],
        regression=REGRESSION,
        max_epochs=N_ITER,
        verbose=1,
    )
    return Net(l, **args)


def optimize_weights(preds, labels):
    print(np.array(preds).shape)
    print(np.array(labels).shape)
    from scipy.optimize import minimize
    def neg_kappa_from_weights(x):
        return - util.kappa(average(x, preds), labels)
    n = len(preds)
    res = minimize(neg_kappa_from_weights, np.ones(n) / n, method='Powell')
    print(res)
    return res.x / np.sum(res.x), -res.fun


def average(w, preds):
    return np.array(w).dot(preds) / np.sum(w)


@click.command()
@click.option('--cnf', default='config/c_512_4x4_very.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
@click.option('--grid_search', is_flag=True, default=False)
@click.option('--per_patient', is_flag=True, default=False)
@click.option('--transform_file', default=None)
@click.option('--directory', default='data/features')
@click.option('--n_iter', default=1)
@click.option('--transform', is_flag=True, default=False)
def fit(cnf, predict, grid_search, per_patient, transform_file, n_iter,
        directory, transform):

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)
    if REGRESSION:
        labels = labels.astype(np.float32)[:, np.newaxis]
    else:
        labels = labels.astype(np.int32)

    #dirs = glob('data/features/*/') if directory is None else [directory]
    dirs = sorted(glob('{}/*/'.format(directory)))
    files = sorted(glob('{}/*.*'.format(directory)))

    if transform_file is None:
        X_trains = [load_transform(directory=directory)
                    for directory in dirs] \
                   + [load_transform(transform_file=transform_file)
                      for transform_file in files]
    else:
        X_trains = [load_transform(transform_file=transform_file)]

    scalers = [StandardScaler() for _ in X_trains]
    X_trains = [scaler.fit_transform(X_train) 
                for scaler, X_train in zip(scalers, X_trains)]

    if predict:

        if transform_file is None:
            X_tests = [load_transform(directory=directory, test=True)
                       for directory in dirs]
        else:
            transform_file = transform_file.replace('train', 'test')
            X_tests = [load_transform(test=True, 
                                      transform_file=transform_file)]

        X_tests = [scaler.transform(X_test) 
                   for scaler, X_test in zip(scalers, X_tests)]

        #if per_patient:
        #    X_tests = [per_patient_reshape(X_test).astype(np.float32)
        #               for X_test in X_tests]

    # util.split_indices split per patient by default now
    tr, te = util.split_indices(labels)

    # 
    log_labels = np.log(labels + 1)

    print("feature matrix {}".format(X_train.shape))

    y_preds = []
    y_preds_train = []
    y_subs = []
    #for d, X_train in zip(dirs, X_trains):
    for X_train, X_test in zip(X_trains, X_tests):
        for i in range(n_iter):
            #print('dir', d)
            print('iter {}'.format(i))

            #a_std = np.argsort(np.std(X_train, 1))
            #X_train[a_std[:1000]] = 0.0

            X = per_patient_reshape(X_train) if per_patient else X_train

            print('fitting split training set')
            est = get_estimator(X.shape[1], randomize=RANDOMIZE)
            est.fit(X, labels)

        
            if predict:
                Xt = per_patient_reshape(X_test) if per_patient else X_test
                p_est = get_estimator(X.shape[1], eval_size=0.0, test=True)
                print('reloading weights from fitted estimator')
                p_est.load_params_from(est)
                p_est.fit(X, labels)
                y_subs.append(p_est.predict(Xt).ravel())

            #if transform:
            #    X_feat = est.transform(X)
            #    fname = os.path.join(
            #        BLEND_FEATURE_DIR,
            #        '{}.npy'.format(directory.split('/')[-1]))

            #    np.save(fname, X_feat)
            #    print("saved transform to {}".format(fname))

            y_pred_train = est.predict(X).ravel()
            y_preds_train.append(y_pred_train)

            y_pred = est.predict(X[te]).ravel()
            #y_pred = est.predict_proba(X[te])
            y_preds.append(y_pred)
            y_pred = np.mean(y_preds, axis=0)
            #y_pred = np.dot(np.mean(y_preds, axis=0), np.arange(5))


            np.save('preds.npy', np.array(y_preds).T)
            np.save('preds_train.npy', np.array(y_preds_train).T)

            # level 4 is very difficult to get right, only require it to 
            # occur in one of the fits
            #from scipy.stats import mode
            #y_pred, _ = mode(np.round(y_preds), axis=0)
            #y_pred = y_pred.squeeze()
            #y_pred = np.mean(np.exp(y_preds), axis=0) - 1
            #y_max = np.max(y_preds, axis=0)
            #y_pred[(y_pred >= 3.0) & (y_max >= 3.75)] = 4.0

            y_pred  = np.clip(np.round(y_pred).astype(int),
                              np.min(labels), np.max(labels))

            #if len(y_preds) >= 2:
            #    w, kappa = optimize_weights(y_preds_train, labels)
            #    print('best weights: ', w)
            #    print('best kappa: ', kappa)
            #    y_pred = average(w, y_preds)
            #    y_pred  = np.clip(np.round(y_pred).astype(int),
            #                      np.min(labels), np.max(labels))
            #    #print('best kappa treshholds', tresh)
            #    #print('best kappa train', kappa)
            #    #y_pred = labels_from_thresholds(tresh, y_pred)

            #    #print(labels[te].ravel().shape, y_pred.shape)
            #    print('kappa', i, util.kappa(labels[te], y_pred))
            #    print(confusion_matrix(labels[te], y_pred))

            #print(labels[te].ravel().shape, y_pred.shape)
            print('kappa', i, util.kappa(labels[te], y_pred))
            print(confusion_matrix(labels[te], y_pred))
        
        #w, kappa = optimize_weights(y_preds, labels[te])
        #print('best weights {}'.format(w))
        #print('best kappa {}'.format(kappa))

    if predict:

        y_preds = y_subs
        #y_preds_train = []
        #for i in range(n_iter):
        #    for X_train, X_test in zip(X_trains, X_tests):
        #        print('fitting full training set')
        #        X = per_patient_reshape(X_train) if per_patient else X_train
        #        Xt = per_patient_reshape(X_test) if per_patient else X_test
        #        est = get_estimator(X.shape[1], eval_size=0.0, 
        #                            randomize=RANDOMIZE)
        #        est.fit(X, labels)
        #        y_pred = est.predict(Xt).ravel()
        #        y_preds.append(y_pred)

        y_pred = np.mean(y_preds, axis=0)
        #y_pred = average(w, y_preds)

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
