from __future__ import division
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier

from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import util
import nn

from definitions import *

TRANSFORM_DIR = 'data/transform_spatial'

@click.command()
@click.option('--cnf', default='config/large.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
@click.option('--grid_search', is_flag=True, default=False)
@click.option('--per_patient', is_flag=True, default=False)
def fit(cnf, predict, grid_search, per_patient):

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)

    tfs = sorted([f for f in os.listdir(TRANSFORM_DIR) if f.endswith('npy')])

    transforms = {'test': [], 'train': []}

    for tf in tfs:
        print(tf)
        t = np.load(open(os.path.join(TRANSFORM_DIR, tf), 'rb'))
        print(t.shape)
        if 'test' in tf:
            transforms['test'].append(t)
        else:
            transforms['train'].append(t)

    X_train = np.hstack([t.reshape([t.shape[0], -1]) for t in
                         transforms['train']])
    if predict:
        X_test = np.hstack([t.reshape([t.shape[0], -1]) for t in
                            transforms['test']])
        y_train = labels

    est = XGBRegressor(
        n_estimators=100, 
        #objective='rank:pairwise',
        silent=0,
        subsample=0.5,
        #colsample_bytree=0.02,
        #max_depth=10,
        #learning_rate=0.05,
    )
    #est = Ridge()
    #est = Pipeline([
    #    #('scale', MinMaxScaler()),
    #    ('fit', LinearSVR(verbose=3, max_iter=10000)),
    #])

    if per_patient:
        X_train = X_train.reshape(len(X_train) / 2, -1).repeat(2, axis=0)
        if not predict:
            a, b = util.split_indices(labels)
            
            even = b[b%2 == 0]
            odd = b[b%2 != 0]
            te = np.array(
                list(set.union(set(even), set(even+1), set(odd), set(odd-1))),
                dtype=int)
            logical = np.zeros(len(labels), dtype=bool)
            logical[te] = True
            tr = np.where(~logical)[0]
            X_train, X_test = X_train[tr], X_train[te]
            y_train, y_test = labels[tr], labels[te]
    else:
        X_train, X_test, y_train, y_test = util.split(X_train, labels)


    if not predict:
        grid = {
            #'subsample': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
            #'colsample_bytree': [0.01, 0.02, 0.05, 0.1],
            #'max_depth': [1, 2, 3],
            #'learning_rate': [0.02, 0.05, 0.08, 0.1],
            #'n_estimators': [100, 200, 500],
            #'fit__epsilon': [0.01, 0.1, 0.4, 0.5],
            #'fit__C': [0.01, 0.1, 1, 10],
            'alpha': [0.001, 0.01, 0.1, 0.2, 0.5, 1, 10],
        }
        print("feature matrix {}".format(X_train.shape))
        indices = util.split_indices(labels)
        kappa_scorer = make_scorer(util.kappa)
        y_train = labels

        if grid_search:
            gs = GridSearchCV(est, grid, verbose=3, cv=[indices], 
                              scoring=kappa_scorer, n_jobs=-1)
            gs.fit(X_train, y_train)
            pd.set_option('display.height', 500)
            pd.set_option('display.max_rows', 500)
            df = util.grid_search_score_dataframe(gs)
            print(df)
            df.to_csv('grid_scores.csv')
            est = gs.best_estimator_
        else:
            est.fit(X_train, y_train)
            y_pred = np.round(est.predict(X_test)).astype(int)
            #y_pred = y_pred.reshape(len(y_pred) * 2, 1)
            print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
            print("kappa {}".format(util.kappa(y_test, y_pred)))

    if predict:
        est.fit(X_train, y_train)
        y_pred = np.round(est.predict(X_test)).astype(int)
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
