from __future__ import division
from collections import Counter
import importlib

import click
import numpy as np

from sklearn.ensemble import *
from quadratic_weighted_kappa import quadratic_weighted_kappa
import util
from util import pickle
import nn

from definitions import *

@click.command()
@click.option('--cnf', default='config/best.py',
              help="Path or name of configuration module.")
@click.option('--balance', default=0.1,
              help="Balancing ratio in [0, 1]. 0:unbalanced, 1:balanced.")
def fit(cnf, balance):

    config = util.load_module(cnf)

    files = util.get_image_files(TRAIN_DIR)
    names = util.get_names(files)
    labels = util.get_labels(names)

    f_train, f_test, y_train, y_test = util.split(files, labels)

    mean = util.get_mean(files)

    #X_train = np.load(TRAIN_FEATURES)
    net = nn.create_net(mean, config.layers)
    net.load_weights_from(WEIGHTS)

    X_train = net.transform(f_train)
    X_test = net.transform(f_test)
    #X_test = np.load(TEST_FEATURES)
    #y_train = np.load(TRAIN_LABELS)
    #y_test = np.load(TEST_LABELS)

    # best so far, score ~ 0.50
    #estimator = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, verbose=2)

    # balance the samples via sample weight (currently leads to worse results)
    counter = Counter(y_train)
    sample_weight = np.zeros(y_train.shape)
    max_count = np.max(counter.values())
    for label, count in counter.items():
        weight = balance / count + (1 - balance) / len(y_train)
        sample_weight[y_train == label] = weight

    #sample_weight += np.mean(sample_weight)

    estimator = GradientBoostingRegressor(verbose=2)
    print("fitting regressor")
    estimator.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = estimator.predict(X_test)
    y_pred_int = np.round(y_pred).astype(int)

    #estimator = GradientBoostingClassifier(n_estimators=100)
    #print("fitting classifier")
    #estimator.fit(X_train, y_train, sample_weight=sample_weight)
    #y_pred = estimator.predict_proba(X_test)
    #y_pred_int = np.round(y_pred.dot(range(y_pred.shape[1])))

    score = quadratic_weighted_kappa(y_test, y_pred_int)
    print("quadratic weighted kappa score {}".format(score))

    pickle.dump(estimator, open(ESTIMATOR_FILENAME, 'wb'))

    print("saved estimator to {}".format(ESTIMATOR_FILENAME))


if __name__ == '__main__':
    fit()
