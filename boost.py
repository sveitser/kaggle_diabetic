from __future__ import division
import os

import click
import numpy as np
from sklearn.metrics import confusion_matrix
from xgboost import XGBRegressor

import util
import nn

from definitions import *

@click.command()
@click.option('--cnf', default='config/large.py',
              help="Path or name of configuration module.")
def fit(cnf):

    model = util.load_module(cnf).model
    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)

    #X = model.load_transform()
    tfs = os.listdir(TRANSFORM_DIR)
    transforms = [np.load(open(os.path.join(TRANSFORM_DIR, tf), 'rb')) 
                  for tf in tfs]
    print([t.shape for t in transforms])
    X = np.hstack([t.reshape([t.shape[0], -1]) for t in transforms])

    X_train, X_test, y_train, y_test = util.split(X, labels)

    est = XGBRegressor(silent=0)
    est.fit(X_train, y_train)
    y_pred = np.round(est.predict(X_test))

    print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
    print("kappa {}".format(util.kappa(y_test, y_pred)))

if __name__ == '__main__':
    fit()
