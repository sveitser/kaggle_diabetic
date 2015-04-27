import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from quadratic_weighted_kappa import quadratic_weighted_kappa
from util import pickle

from definitions import *


def fit():

    X_train = np.load(TRAIN_FEATURES)
    X_test = np.load(TEST_FEATURES)
    y_train = np.load(TRAIN_LABELS)
    y_test = np.load(TEST_LABELS)

    # best so far, score ~ 0.41
    estimator = GradientBoostingRegressor(verbose=2) 

    print("fitting regressor")
    y_pred = estimator.fit(X_train, y_train).predict(X_test)
    y_pred_int = np.round(y_pred).astype(int)

    score = quadratic_weighted_kappa(y_test, y_pred_int)
    print("quadratic weighted kappa score {}".format(score))

    pickle.dump(estimator, open(ESTIMATOR_FILENAME, 'wb'))

    print("saved estimator to {}".format(ESTIMATOR_FILENAME))


if __name__ == '__main__':
    fit()
