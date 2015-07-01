from __future__ import division
import importlib

import click
import numpy as np

import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu1") # doesn't seem to work when set here
#import theano

import util
import nn

from definitions import *
from tta import build_quasirandom_transforms


@click.command()
@click.option('--cnf', default='config/best.py',
              help="Path or name of configuration module.")
@click.option('--n_iter', default=1,
              help="Iterations for test time averaging.")
@click.option('--test', is_flag=True, default=False)
@click.option('--train', is_flag=True, default=False)
@click.option('--weights_from', default=None,
              help='Path to initial weights file.', type=str)
def transform(cnf, n_iter, test, train, weights_from):

    model = util.load_module(cnf).model

    runs = {}
    if train:
        runs['train'] = model.get('train_dir', TRAIN_DIR)
    if test:
        runs['test'] = model.get('test_dir', TEST_DIR)

    model.cnf['batch_size_test'] = 32

    net = nn.create_net(model, tta=True if n_iter > 1 else False)

    if weights_from is None:
        net.load_params_from(model.weights_file)
        print("loaded weights from {}".format(model.weights_file))
    else:
        weights_from = str(weights_from)
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))

    tfs = build_quasirandom_transforms(n_iter, **model.cnf['aug_params'])

    for run, directory in sorted(runs.items(), reverse=True):

        print("transforming {}".format(directory))

        files = util.get_image_files(directory)

        X_t = None

        for i, tf in enumerate(tfs):

            print("{} transform iter {}".format(run, i + 1))

            if X_t is None:
                X_t = net.transform(files, transform=tf)
            else:
                X_t += net.transform(files, transform=tf)

            if i % 5 == 4 or n_iter < 5:
                model.save_transform(X_t / (n_iter + 1), i + 1,
                                     test=True if run == 'test' else False)
                print('saved {} iterations'.format(i + 1))

if __name__ == '__main__':
    transform()
