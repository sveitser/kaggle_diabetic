from __future__ import division
from collections import Counter
from datetime import datetime
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier

from sklearn import linear_model
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import util
import nn

from definitions import *

#TRANSFORM_DIR = 'data/transform_spatial'


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
    print(left_idx)
    print(right_idx)
    return np.hstack([X[left_idx], X[right_idx], right_eye])


def get_sample_weights(y):
    c = Counter(y)
    f = np.vectorize(lambda x: 1.0 / c[x])
    w = f(y)
    w *= len(y) / float(np.sum(w))
    return (2.0 + w) / 3.0


def load_transform(directory=TRANSFORM_DIR, test=False, transform_file=None):

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

@click.command()
@click.option('--cnf', default='config/large.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
@click.option('--grid_search', is_flag=True, default=False)
@click.option('--per_patient', is_flag=True, default=False)
@click.option('--transform_file', default=None)
def fit(cnf, predict, grid_search, per_patient, transform_file):

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)

    X_train = load_transform(transform_file=transform_file)


    if per_patient:
        X_train = per_patient_reshape(X_train)

    if predict:
        X_test = load_transform(test=True)
        y_train = labels
        if per_patient:
            X_test = per_patient_reshape(X_test)
    else:
        if per_patient:
            tr, te = per_patient_split(labels)
        else:
            tr, te = util.split_indices(labels)

    #print('doing PCA')
    #from sklearn.decomposition import PCA, RandomizedPCA
    #pca = RandomizedPCA(n_components=500)
    #X_train = pca.fit_transform(X_train)


    est = XGBRegressor(
        n_estimators=100, 
        #objective='rank:pairwise',
        silent=0,
        subsample=0.5,
        colsample_bytree=0.15,
        max_depth=3,
        learning_rate=0.1,
        seed=42,
    )

    #est = LinearSVC(verbose=2, class_weight='auto')
    #est = ExtraTreesRegressor(100, n_jobs=-1, verbose=2,
    #                          max_leaf_nodes=100)

    #est = Pipeline([
    #    ('scale', MinMaxScaler()),
    ##    #('fit', LinearSVR(C=1, epsilon=0.1, verbose=3, max_iter=10000)),
    ##    ('fit', linear_model.Ridge()),
    #     ('fit', linear_model.LogisticRegression(verbose=2,
    #                                             class_weight='auto', C=0.1))
    #])


    if not predict:
        grid = {
            #'subsample': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            #'colsample_bytree': [0.05, 0.1, 0.15],
            #'max_depth': [3, 4, 5],
            #'learning_rate': [0.05, 0.1, 0.15],
            #'n_estimators': [50, 100, 150],
            #'fit__epsilon': [0.05, 0.1, 0.2, 0.25, 0.3],
            #'fit__C': [1.0, 2.0, 5.0, 10.0, 100.0],
            'fit__alpha': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 10.0],
        }
        print("feature matrix {}".format(X_train.shape))

        if grid_search:
            kappa_scorer = make_scorer(util.kappa)
            y_train = labels
            gs = GridSearchCV(est, grid, verbose=3, cv=[(tr, te)], 
                              scoring=kappa_scorer, n_jobs=4)
            gs.fit(X_train, y_train)
            pd.set_option('display.height', 500)
            pd.set_option('display.max_rows', 500)
            df = util.grid_search_score_dataframe(gs)
            print(df)
            df.to_csv('grid_scores_{}.csv'.format(datetime.now().isoformat()))
            est = gs.best_estimator_
        else:
            est.fit(X_train[tr], labels[tr])
                    #sample_weight=get_sample_weights(labels[tr]))
            y_pred = np.round(est.predict(X_train[te])).astype(int)
            y_pred = np.clip(y_pred, np.min(labels), np.max(labels))
            #y_pred = y_pred.reshape(len(y_pred) * 2, 1)
            print(confusion_matrix(labels[te], y_pred))
            print("kappa {}".format(util.kappa(labels[te], y_pred)))

    if predict:
        est.fit(X_train, y_train)

        y_pred = np.round(est.predict(X_test)).astype(int)
        y_pred = np.clip(y_pred, np.min(y_train), np.max(y_train))
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
