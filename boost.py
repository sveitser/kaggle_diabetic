from __future__ import division
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from xgboost import XGBRegressor

import util
import nn

from definitions import *

@click.command()
@click.option('--cnf', default='config/large.py',
              help="Path or name of configuration module.")
@click.option('--predict', is_flag=True, default=False)
def fit(cnf, predict):

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)

    tfs = sorted(os.listdir(TRANSFORM_DIR))

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
    else:
        X_train, X_test, y_train, y_test = util.split(X_train, labels)

    est = XGBRegressor(
        n_estimators=100, 
        #silent=0,
        #subsample=0.5,
        #colsample_bytree=0.5,
    )
    grid = {
        'subsample': [0.1, 0.5, 0.8, 1.0],
        'colsample_bytree': [0.1, 0.5, 0.8, 1.0],
        'max_depth': [1, 2, 3, 4, 5, 6],
        'n_estimators': [50, 100, 200],
    }
    gs = GridSearchCV(est, grid, verbose=2)

    gs.fit(X_train, y_train)
    pd.set_option('display.height', 500)
    pd.set_option('display.max_rows', 500)
    df = util.grid_search_score_dataframe(gs)
    print(df)
    df.to_csv('grid_scores.csv')

    y_pred = np.round(gs.predict(X_test)).astype(int)

    if predict:
        submission_filename = util.get_submission_filename()
        files = util.get_image_files(model.get('test_dir', TEST_DIR))
        names = util.get_names(files)
        image_column = pd.Series(names, name='image')
        level_column = pd.Series(y_pred, name='level')
        predictions = pd.concat([image_column, level_column], axis=1)
        predictions.to_csv(submission_filename, index=False)
        print("saved predictions to {}".format(submission_filename))
    else:
        print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
        print("kappa {}".format(util.kappa(y_test, y_pred)))

if __name__ == '__main__':
    fit()
