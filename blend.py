"""Blend features extracted with Conv Nets and make predictions/submissions."""
from __future__ import division, print_function
from datetime import datetime
from glob import glob

import click
import numpy as np
import pandas as pd
import theano
from lasagne import init
from lasagne.updates import adam
from lasagne.nonlinearities import rectify
from lasagne.layers import DenseLayer, InputLayer, FeaturePoolLayer
from nolearn.lasagne import BatchIterator
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import yaml

import data
import nn
import util

np.random.seed(9)

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


class BlendNet(nn.Net):

    def set_split(self, files, labels):
        """Override train/test split method to use our default split."""
        def split(X, y, eval_size):
            if eval_size:
                tr, te = data.split_indices(files, labels, eval_size)
                return X[tr], X[te], y[tr], y[te]
            else:
                return X, X[len(X):], y, y[len(y):]
        setattr(self, 'train_test_split', split)


class ResampleIterator(BatchIterator):

    def __init__(self, batch_size, resample_prob=0.2, shuffle_prob=0.5):
        self.resample_prob = resample_prob
        self.shuffle_prob = shuffle_prob
        super(ResampleIterator, self).__init__(batch_size)

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        indices = data.balance_per_class_indices(self.y.ravel())
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


def get_estimator(n_features, files, labels, eval_size=0.1):
    layers = [
        (InputLayer, {'shape': (None, n_features)}),
        (DenseLayer, {'num_units': N_HIDDEN_1, 'nonlinearity': rectify,
                      'W': init.Orthogonal('relu'),
                      'b': init.Constant(0.01)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': N_HIDDEN_2, 'nonlinearity': rectify,
                      'W': init.Orthogonal('relu'),
                      'b': init.Constant(0.01)}),
        (FeaturePoolLayer, {'pool_size': 2}),
        (DenseLayer, {'num_units': 1, 'nonlinearity': None}),
    ]
    args = dict(
        update=adam,
        update_learning_rate=theano.shared(util.float32(START_LR)),
        batch_iterator_train=ResampleIterator(BATCH_SIZE),
        batch_iterator_test=BatchIterator(BATCH_SIZE),
        objective=nn.get_objective(l1=L1, l2=L2),
        eval_size=eval_size,
        custom_score=('kappa', util.kappa) if eval_size > 0.0 else None,
        on_epoch_finished=[
            nn.Schedule('update_learning_rate', SCHEDULE),
        ],
        regression=True,
        max_epochs=N_ITER,
        verbose=1,
    )
    net = BlendNet(layers, **args)
    net.set_split(files, labels)
    return net


@click.command()
@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False, show_default=True,
              help="Make predictions on test set features after training.")
@click.option('--per_patient', is_flag=True, default=False, show_default=True,
              help="Blend features of both patient eyes.")
@click.option('--features_file', default=None, show_default=True,
              help="Read features from specified file.")
@click.option('--n_iter', default=1, show_default=True,
              help="Number of times to fit and average.")
@click.option('--blend_cnf', default='blend.yml', show_default=True,
              help="Blending configuration file.")
@click.option('--test_dir', default=None, show_default=True,
              help="Override directory with test set images.")
def fit(cnf, predict, per_patient, features_file, n_iter, blend_cnf, test_dir):

    config = util.load_module(cnf).config
    image_files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(image_files)
    labels = data.get_labels(names).astype(np.float32)[:, np.newaxis]

    if features_file is not None:
        runs = {'run': [features_file]}
    else:
        runs = data.parse_blend_config(yaml.load(open(blend_cnf)))

    scalers = {run: StandardScaler() for run in runs}

    tr, te = data.split_indices(image_files, labels)

    y_preds = []
    for i in range(n_iter):
        print("iteration {} / {}".format(i + 1, n_iter))
        for run, files in runs.items():
            print("fitting features for run {}".format(run))
            X = data.load_features(files)
            X = scalers[run].fit_transform(X)
            X = data.per_patient_reshape(X) if per_patient else X
            est = get_estimator(X.shape[1], image_files, labels,
                                eval_size=0.0 if predict else 0.1)
            est.fit(X, labels)
            if not predict:
                y_pred = est.predict(X[te]).ravel()
                y_preds.append(y_pred)
                y_pred = np.mean(y_preds, axis=0)
                y_pred = np.clip(np.round(y_pred).astype(int),
                                 np.min(labels), np.max(labels))
                print("kappa after run {}, iter {}: {}".format(
                    run, i, util.kappa(labels[te], y_pred)))
                print("confusion matrix")
                print(confusion_matrix(labels[te], y_pred))
            else:
                X = data.load_features(files, test=True)
                X = scalers[run].transform(X)
                X = data.per_patient_reshape(X) if per_patient else X
                y_pred = est.predict(X).ravel()
                y_preds.append(y_pred)

    if predict:
        y_pred = np.mean(y_preds, axis=0)
        y_pred = np.clip(np.round(y_pred),
                         np.min(labels), np.max(labels)).astype(int)
        submission_filename = util.get_submission_filename()
        image_files = data.get_image_files(test_dir or config.get('test_dir'))
        names = data.get_names(image_files)
        image_column = pd.Series(names, name='image')
        level_column = pd.Series(y_pred, name='level')
        predictions = pd.concat([image_column, level_column], axis=1)

        print("tail of predictions file")
        print(predictions.tail())

        predictions.to_csv(submission_filename, index=False)
        print("saved predictions to {}".format(submission_filename))


if __name__ == '__main__':
    fit()
