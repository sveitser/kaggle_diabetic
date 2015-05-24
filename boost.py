from __future__ import division, print_function
from collections import Counter
from datetime import datetime
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
from scipy.optimize import minimize

from lightning.ranking import PRank, KernelPRank

from sklearn import linear_model
from sklearn.ensemble import *
from sklearn.svm import *
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
    #args = {
    #    'n_estimators': np.random.choice([90, 100, 110]),
    #    'subsample': np.random.choice([0.3, 0.5, 0.7]),
    #    'colsample_bytree': np.random.choice([0.01, 0.02, 0.05]),
    #    'learning_rate': np.random.choice([0.08, 0.09, 0.1]),
    #    'seed': np.random.randint(100),
    #}
    args = {
        'subsample': 0.5,
        #'colsample_bytree': 0.005,
        'colsample_bytree': 0.1,
        #'colsample_bytree': 0.001,
        'learning_rate': 0.1,
        'seed': 42,
        'n_estimators': 100,
    }
    args.update(kwargs)
    import pprint
    pprint.pprint(args)
    return XGBRegressor(**args)

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


def per_patient_reshape(X):
    right_eye = np.arange(0, X.shape[0])[:, np.newaxis] % 2
    #X = X.reshape(len(X) / 2, -1).repeat(2, axis=0)
    n = len(X)
    left_idx = np.arange(n)
    right_idx = left_idx + np.sign(2 * ((left_idx + 1) % 2) - 1)

    return np.hstack([X[left_idx], X[right_idx], right_eye])

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

    data = [np.load(open(tf, 'rb')) for tf in tfs]

    return np.hstack([t.reshape([t.shape[0], -1]) for t in data])

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

    #if per_patient:
    #    tr, te = per_patient_split(labels)
    #else:
    #    tr, te = util.split_indices(labels)
        
    # util.split_indices split per patient by default now
    tr, te = util.split_indices(labels)

    # TODO remove these 2 lines
    #n = len(labels)
    #tr = np.hstack([tr, n + tr, 2*n + tr])
    #te = np.hstack([te, n + te, 2*n + te])
    #labels = np.tile(labels, 3)
    #print(tr, te)

    est = get_xgb()
    #est = linear_model.Ridge()
    #est = LinearSVR(verbose=2)
    #est = OrdinalClassifier(XGBClassifier(silent=0, colsample_bytree=0.1,
    #                                      subsample=0.5))
    #est = make_pipeline(StandardScaler(), LinearSVR(verbose=3, max_iter=10000))
    #
    #   TODO: sort and threshold according to original distribution
    # 
    if not predict:
        grid = {
            #'subsample': [0.5, 1.0],
            'colsample_bytree': [0.005, 0.01, 0.02, 0.05, 0.1,
                                 0.2],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.08, 0.1, 0.12],
            'n_estimators': [80, 100, 120],
            'seed': np.arange(n_iter) * 9,
            #'n_estimators': [50, 100, 200],
            #'epsilon': [0.1, 0.2, 0.25, 0.3],
            #'C': [5.0, 10, 20],
            #'alpha': [0.002, 0.005, 0.01, 0.02, 0.05],
            #'alpha': [0.1, 0.2, 0.5, 1.0, 2.0],
            #'alpha': [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            #'fit__C': [0.01, 0.1, 1.0, 10, 100],
        }
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
                #est = get_xgb(seed=i*10)
                est.fit(X_train[tr], labels[tr])
                y_pred = est.predict(X_train[te])
                y_preds.append(y_pred)
             
                y_pred = np.mean(y_preds, axis=0)
                y_pred  = np.clip(np.round(y_pred).astype(int),
                                  np.min(labels), np.max(labels))

                print('kappa', i, util.kappa(labels[te], y_pred))
                print(confusion_matrix(labels[te], y_pred))

    if predict:

        print("fitting validation set for thresholds")

        y_preds = []
        for i in range(n_iter):
            print('fitting full training set')
            est = get_xgb(seed=i * 10)
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
