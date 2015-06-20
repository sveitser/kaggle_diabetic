from __future__ import division, print_function
from collections import Counter
from datetime import datetime
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.grid_search import GridSearchCV
import xgboost as xgb


from lightning.ranking import PRank, KernelPRank

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ordinal_classifier import OrdinalClassifier
import util
import nn

from definitions import *
from boost import *

def evalkappa(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', - util.kappa(labels, np.clip(np.round(preds), np.min(labels), 
                                                 np.max(labels)))

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

    # util.split_indices split per patient by default now
    tr, te = util.split_indices(labels)

    dtrain = xgb.DMatrix(X_train[tr], label=labels[tr])
    dtest = xgb.DMatrix(X_train[te], label=labels[te])

    param = {
        'silent': 1,
        'eta': 0.1,
        'objective':'rank:pairwise',
        'booster':'gbtree',
        'colsample_bytree': 0.001,
        #'alpha': 0.01,
        #'lambda': 10.0,
        'max_depth': 10,
        'subsample': 0.5,
    }

    watchlist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 100
    bst = xgb.train(param, dtrain, num_round, watchlist, feval=evalkappa)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    print(labels)
    exit(0)

    est = get_estimator()
    # 
    if not predict:
        grid = {
            #'subsample': [0.5, 1.0],
            'colsample_bytree': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
            #'colsample_bytree': [0.002],
            #'max_depth': [2, 3, 4],
            #'learning_rate': [0.05],
            #'n_estimators': [500],
            #'seed': np.arange(n_iter) * 9,
            #'n_estimators': [50, 100, 200],
            #'epsilon': [0.1, 0.2, 0.25, 0.3],
            #'C': [5.0, 10, 20],
            #'alpha': [0.002, 0.005, 0.01, 0.02, 0.05],
            #'alpha': [0.1, 0.2, 0.5, 1.0, 2.0, 5, 10],
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
                est = get_estimator(seed=i * 10 + 1)
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
            est = get_estimator(seed=i * 10 + 1)
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
