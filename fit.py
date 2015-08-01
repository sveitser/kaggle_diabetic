from __future__ import division
from collections import Counter
import importlib

import click
import numpy as np

from sklearn.ensemble import *
from sklearn.svm import *
from sklearn import linear_config
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from quadratic_weighted_kappa import quadratic_weighted_kappa
import util
from util import pickle

from definitions import *

from yart import OrdinalRegression

from scipy.optimize import minimize

def eval(w, y_true, y_pred):
    error = - util.kappa(y_true, np.round(y_pred.dot(w)).astype(int))
    print(-error)
    return error

@click.command()
@click.option('--balance', default=0.1,
              help="Balancing ratio in [0, 1]. 0:unbalanced, 1:balanced.")
def fit(balance):

    files = util.get_image_files(TRAIN_DIR)
    names = util.get_names(files)
    labels = util.get_labels(names)

    f_train, f_test, y_train, y_test = util.split(files, labels)

    n_train = len(f_train)
    n_test = len(f_test)
    X_train = np.hstack(np.load(TRAIN_FEATURES).reshape(10, n_train, -1))
    X_test = np.hstack(np.load(TEST_FEATURES).reshape(10, n_test, -1))

    print(X_train.shape)

    #y_train = np.tile(y_train, 10)

    #X_train = X_train.reshape(10, -1, *X_train.shape[1:]).mean(axis=0)
    #X_test = X_test.reshape(10, -1, *X_test.shape[1:]).mean(axis=0)

    #print X_train.shape
    #print X_test.shape

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

    #estimator = ExtraTreesClassifier(n_estimators=200, verbose=2, n_jobs=-1)
    #estimator = ExtraTreesClassifier(n_estimators=200, verbose=2, n_jobs=-1)
    #estimator = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=-1)
    #estimator = GradientBoostingClassifier(n_estimators=100, verbose=2,
    #                                       max_features=50,
    #                                       learning_rate=0.05,
    #                                       subsample=0.5, max_depth=3)
    estimator = LinearSVC(verbose=2)
    #estimator = GradientBoostingRegressor(verbose=2, max_features=500,
    #                                      subsample=0.5)

    #estimator = make_pipeline(
    #    StandardScaler(with_mean=False),
    #    SVC(verbose=2, probability=True),
    #)
    print("fitting regressor")
    #estimator.fit(X_train, y_train, svc__sample_weight=sample_weight)
    #from scipy.sparse import csr_matrix
    #estimator.fit(csr_matrix(X_train.astype(np.float)), y_train.astype(int))
    #estimator.fit(X_train, y_train, sample_weight=sample_weight)
    estimator.fit(X_train, y_train)
    #y_pred = estimator.predict_proba(X_test)
    #res = minimize(eval, np.arange(5), args=(y_test, y_pred),
    #               method='Powell')
    #print(res)
    y_pred = estimator.predict(X_test)

    #y_pred = y_pred.reshape(10, -1).mean(0)

    y_pred_int = np.round(y_pred).astype(int)
    #y_pred_int = np.round(y_pred.dot(res.x)).astype(int)

    #estimator = GradientBoostingClassifier(n_estimators=20, verbose=2)
    #estimator = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, verbose=2)
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
