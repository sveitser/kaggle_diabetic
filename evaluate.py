from __future__ import division
import importlib

import click
import numpy as np
from sklearn.metrics import confusion_matrix

import util
import nn

from definitions import *

@click.command()
@click.option('--cnf', default='config/best.py',
              help="Path or name of configuration module.")
@click.option('--weights', default=None,
              help="Path to weights file.", type=str)
def fit(cnf, weights):

    # load params doesn't work with unicode
    if weights is not None:
        weights = str(weights)

    model = util.load_module(cnf).model

    model.cnf['batch_size'] = 8

    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))
    names = util.get_names(files)
    labels = util.get_labels(names)

    f_train, f_test, y_train, y_test = util.split(files, labels)

    net = nn.create_net(model, tta=False)
    if weights is None:
        net.load_params_from(model.weights_file)
    else:
        net.load_params_from(weights)

    #ua_train = net.predict(f_train).reshape(10, -1).mean(axis=0)
    preds = []
    for i in range(1):
        print ("predicting {}".format(i))
        preds.append(net.predict(f_test).flatten())

    y_pred = np.round(np.array(preds).mean(axis=0))

    print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))

    print("kappa {}".format(util.kappa(y_test, y_pred)))

if __name__ == '__main__':
    fit()
