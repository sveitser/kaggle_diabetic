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
from xgboost import XGBRegressor, XGBClassifier
from scipy.optimize import minimize

from sklearn import linear_model
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import calibration

from ordinal_classifier import OrdinalClassifier
import util
import nn

from definitions import *

#TRANSFORM_DIR = 'data/transform_spatial'
#FEATURE_DIR = 'data/tta'

def get_xgb(**kwargs):
    grid = {
        #'colsample_bytree': [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
        #                     0.05],
        'colsample_bytree': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        #'colsample_bytree': [0.1, 0.2, 0.3, 0.5],
        #'colsample_bytree': [0.1, 0.2, 0.5],
        #'max_depth': [2, 3, 4],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'seed': np.arange(kwargs.pop('n_iter', 1)) * 10 + 1,
    }
    args = {
        'subsample': 0.5,
        'colsample_bytree': 0.2,
        'learning_rate': 0.1,
        'seed': 99,
        'n_estimators': 100,
        'max_depth': 3,
        #'silent': False,
    }
    args.update(kwargs)
    pprint.pprint(args)
    p = Pipeline([
        ('scale', StandardScaler()),
        ('fit', XGBRegressor(**args))
    ])
    return p, {'fit__' + k: v for k, v in grid.items()}

def get_ridge(**kwargs):
    return linear_model.Ridge(**kwargs)

def get_lda(**kwargs):
    return make_pipeline(LDA(n_components=100), get_xgb())

def get_svr(**kwargs):
    grid = {
        'fit__C': [5.0, 10.0, 20.0],
        'fit__epsilon': [0.15, 0.2, 0.25],
    }
    p = Pipeline([
        ('tf', PCA(n_components=200)),
        ('fit', SVR(verbose=2)),
    ])
    return p, grid


def per_patient_split(labels):
    a, b = util.split_indices(labels)
    even = b[b%2 == 0]
    odd = b[b%2 != 0]
    te = np.array(list(set.union(set(even), set(even+1), set(odd), 
                                 set(odd-1))), dtype=int)
    logical = np.zeros(len(labels), dtype=bool)
    logical[te] = True
    tr = np.where(~logical)[0]
    return tr, te


def per_patient_reshape(X, X_other=None):
    X_other = X if X_other is None else X_other
    right_eye = np.arange(0, X.shape[0])[:, np.newaxis] % 2
    #X = X.reshape(len(X) / 2, -1).repeat(2, axis=0)
    n = len(X)
    left_idx = np.arange(n)
    right_idx = left_idx + np.sign(2 * ((left_idx + 1) % 2) - 1)

    return np.hstack([X[left_idx], X_other[right_idx], 
                      right_eye]).astype(np.float32)

#def per_patient_reshape(X):
#    right_eye = np.arange(0, X.shape[0])[:, np.newaxis] % 2
#    #X = X.reshape(len(X) / 2, -1).repeat(2, axis=0)
#    n = len(X)
#    left = np.repeat(X[0::2], 2, axis=0)
#    right = np.repeat(X[1::2], 2, axis=0)
#    print(left.shape)
#    print(right.shape)
#    return np.hstack([left, right, right_eye])

def get_sample_weights(y):
    c = Counter(y)
    f = np.vectorize(lambda x: 1.0 / c[x])
    w = f(y)
    w *= len(y) / float(np.sum(w))
    return (2.0 + w) / 3.0


def load_transform(directory=FEATURE_DIR, test=False, transform_file=None):

    if transform_file is None:
        tfs = sorted([os.path.join(directory, f) 
                      for f in os.listdir(directory) if f.endswith('npy')])
    else:
        tfs = [transform_file]

    if test:
        tfs = [tf for tf in tfs if 'test' in tf]
    else:
        tfs = [tf for tf in tfs if 'test' not in tf]

    print('loading transform files')
    pprint.pprint(tfs)

    data = [np.load(open(tf, 'rb')) for tf in tfs]
    data = [t.reshape([t.shape[0], -1]) for t in data]
    #data = [PCA(n_components=100).fit_transform(x) for x in data]
    return np.hstack(data)

    #t0 = data[0]
    #t1 = 2*data[1] - data[0]
    #t2 = 3*data[2] - 2*t1
    #print(t0.mean(), t1.mean(), t2.mean())
    ##data = np.concatenate([data[0][np.newaxis, ...], np.diff(data, axis=0)], axis=0)
    #data = np.vstack([t0, t1, t2])
    #print(data.shape)
    #data = data.reshape([data.shape[0]*data.shape[1], data.shape[2],
    #                     data.shape[3], data.shape[4]])
    #print(data.shape)
    #return data.reshape(data.shape[0], -1)


def rescale_labels(y_pred_f, y_train, alpha=0.5):
    """Setting prediction thresholds to make distributions match.
    
    Turns out this leads to slightly worse results.
    """
    n_test = len(y_pred_f)
    n_train = len(y_train)
    y_pred_s = np.sort(y_pred_f)
    pred_labels = np.zeros(n_test, dtype=int)
    true_dist = Counter(y_train)
    upper_bound = None
    for i in sorted(true_dist):
        if i == 0:
            start = 0
        else:
            start = end
        end = start + int(true_dist[i] * n_test / n_train)
        lower_bound = -np.inf if upper_bound is None else upper_bound
        if i == 4:
            upper_bound = np.inf
        else:
            upper_bound = alpha * y_pred_s[end] + (1 - alpha) * (i + 0.5)
        pred_labels[(y_pred_f >= lower_bound) & (y_pred_f < upper_bound)] = i

    print(pred_labels)
    return pred_labels


def labels_from_thresholds(thresholds, y_pred):
    threshols = np.abs(thresholds)
    pred_labels = np.zeros(y_pred.shape, dtype=int)
    upper = thresholds[0]
    pred_labels[y_pred < upper] = 0
    for i, gap in enumerate(thresholds[1:], start=1):
        lower = upper
        upper = upper + gap
        pred_labels[(y_pred >= lower) & (y_pred < upper)] = i

    pred_labels[y_pred >= upper] = 4

    return pred_labels


def neg_kappa_from_thresholds(thresholds, y_pred, y_true):
    return - util.kappa(labels_from_thresholds(thresholds, y_pred), y_true)


def get_best_thresholds(y_pred, y_true):
    res = minimize(neg_kappa_from_thresholds, 
                   [0.5, 1, 1, 1], (y_pred, y_true),
                   method='Powell')
    return res.x, -res.fun


def average_thresholds(thresholds):
    t = np.array(np.abs(thresholds))
    real_threshold = np.cumsum(t, axis=1)
    print(real_threshold)
    mean_threshold = np.mean(real_threshold, axis=0)
    return [mean_threshold[0]] + list(np.diff(mean_threshold))


@click.command()
@click.option('--cnf', default='config/c_512_4x4_32.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
@click.option('--grid_search', is_flag=True, default=False)
@click.option('--per_patient', is_flag=True, default=False)
@click.option('--transform_file', default=None)
@click.option('--n_iter', default=1)
@click.option('--estimator', default='xgb', 
              type=click.Choice(['xgb', 'svr']))
def fit(cnf, predict, grid_search, per_patient, transform_file, n_iter, 
        estimator):

    if estimator == 'xgb':
        get_estimator = get_xgb
        n_jobs = 1
    elif estimator == 'svr':
        get_estimator = get_svr
        n_jobs = 2

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)

    X_train = load_transform(transform_file=transform_file)


    if per_patient:
        X_train = per_patient_reshape(X_train)

    if predict:

        if transform_file is not None:
            transform_file = transform_file.replace('train', 'test')
        X_test = load_transform(test=True, transform_file=transform_file)

        if per_patient:
            X_test = per_patient_reshape(X_test)

    tr, te = util.split_indices(labels)

    est, grid = get_estimator(n_iter=n_iter)

    if not predict:
        print("feature matrix {}".format(X_train.shape))

        if grid_search:
            print(est)
            print(grid)
            kappa_scorer = make_scorer(util.kappa)
            gs = GridSearchCV(est, grid, verbose=3, cv=[(tr, te)], 
                              scoring=kappa_scorer, n_jobs=n_jobs, refit=False)
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
                est, _ = get_estimator(seed=i * 10 + 99)
                est.fit(X_train[tr], labels[tr])
                y_pred = est.predict(X_train[te])
                y_preds.append(y_pred)
             
                y_pred = np.mean(y_preds, axis=0)
                y_pred  = np.clip(np.round(y_pred).astype(int),
                                  np.min(labels), np.max(labels))

                print('kappa', i, util.kappa(labels[te], y_pred))
                print(confusion_matrix(labels[te], y_pred))

    if predict:

        y_preds = []
        for i in range(n_iter):
            print('fitting full training set')
            est, _ = get_estimator(seed=i * 10 + 1)
            est.fit(X_train, labels)
            y_pred = est.predict(X_test)
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
