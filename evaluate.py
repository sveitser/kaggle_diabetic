from __future__ import division
import importlib

import click
import numpy as np
from sklearn.metrics import confusion_matrix

import util
import nn


@click.command()
@click.option('--cnf', default='config/best.py',
              help="Path or name of configuration module.")
@click.option('--weights', default=None,
              help="Path to weights file.", type=str)
def fit(cnf, weights):

    # load params doesn't work with unicode
    if weights is not None:
        weights = str(weights)

    config = util.load_module(cnf).config

    config.cnf['batch_size'] = 128

    files = data.get_image_files(config.get('train_dir', TRAIN_DIR))
    names = data.get_names(files)
    labels = data.get_labels(names)

    f_train, f_test, y_train, y_test = data.split(files, labels)

    net = nn.create_net(config, tta=False)
    if weights is None:
        net.load_params_from(config.weights_file)
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
