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
@click.option('--n_iter', default=20,
              help="Iterations for test time averaging.")
@click.option('--test', is_flag=True, default=False)
@click.option('--train', is_flag=True, default=False)
def transform(cnf, n_iter, test, train):

    model = util.load_module(cnf).model

    runs = {}
    if train:
        runs['train'] = model.get('train_dir', TRAIN_DIR)
    if test:
        runs['test'] = model.get('test_dir', TEST_DIR)

    model.cnf['batch_size_test'] = 64

    net = nn.create_net(model, tta=True)
    net.load_params_from(model.weights_file)

    for run, directory in runs.items():

        print("transforming {}".format(directory))

        files = util.get_image_files(directory)

        X_t = None

        for i in range(n_iter):

            print("transform iter {}".format(i))

            if X_t is None:
                X_t = net.transform(files)
            else:
                X_t += net.transform(files)

        model.save_transform(X_t / n_iter, 
                             test=True if run == 'test' else False)

if __name__ == '__main__':
    transform()
