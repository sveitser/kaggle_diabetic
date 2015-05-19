from __future__ import division
import importlib

import click
import numpy as np

import util
import nn

from definitions import *

@click.command()
@click.option('--cnf', default='config/best.py',
              help="Path or name of configuration module.")
def fit(cnf):

    config = util.load_module(cnf)

    files = util.get_image_files(TRAIN_DIR)
    names = util.get_names(files)
    labels = util.get_labels(names)

    f_train, f_test, y_train, y_test = util.split(files, labels)

    mean = util.get_mean(files)

    net = nn.create_net(mean, config.layers)
    net.load_params_from(WEIGHTS)

    X_train = net.transform(f_train).reshape(10, *X_train.shape)
    X_test = net.transform(f_test).reshape(10, *X_train.shape)

    X_train /= 10.0
    X_test /= 10.0

    np.save(open(TRAIN_FEATURES, 'wb'), X_train)
    np.save(open(TEST_FEATURES, 'wb'), X_test)


if __name__ == '__main__':
    fit()
