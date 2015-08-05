from __future__ import division
import time

import click
import numpy as np

import nn
import data
import tta
import util

@click.command()
@click.option('--cnf', default='config/c_512_4x4_32.py', show_default=True,
              help="Path or name of configuration module.")
@click.option('--n_iter', default=1, show_default=True,
              help="Iterations for test time averaging.")
@click.option('--skip', default=0, show_default=True,
              help="Number of test time averaging iterations to skip.")
@click.option('--test', is_flag=True, default=False, show_default=True,
              help="Extract features for test set. Ignored if --train_dir is "
                   "specified.")
@click.option('--train', is_flag=True, default=False, show_default=True,
              help="Extract features for test set. Ignored if --test_dir is "
                   "specified.")
@click.option('--weights_from', default=None, show_default=True,
              help='Path to weights file.', type=str)
@click.option('--test_dir', default=None, show_default=True,
              help="Override directory with test set images.")
def transform(cnf, n_iter, skip, test, train, weights_from,  test_dir):

    config = util.load_module(cnf).config

    runs = {}
    if train or train_dir:
        runs['train'] = config.get('train_dir')
    if test or test_dir:
        runs['test'] = test_dir or config.get('test_dir')

    net = nn.create_net(config)

    if weights_from is None:
        net.load_params_from(config.weights_file)
        print("loaded weights from {}".format(config.weights_file))
    else:
        weights_from = str(weights_from)
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))

    if n_iter > 1:
        tfs, color_vecs = tta.build_quasirandom_transforms(
                n_iter, skip=skip, color_sigma=config.cnf['sigma'],
                **config.cnf['aug_params'])
    else:
        tfs, color_vecs = tta.build_quasirandom_transforms(
               n_iter, skip=skip, color_sigma=0.0,
                **data.no_augmentation_params)

    for run, directory in sorted(runs.items(), reverse=True):

        print("extracting features for files in {}".format(directory))
        tic = time.time()

        files = data.get_image_files(directory)

        Xs, Xs2 = None, None

        for i, (tf, color_vec) in enumerate(zip(tfs, color_vecs), start=1):

            print("{} transform iter {}".format(run, i))

            X = net.transform(files, transform=tf, color_vec=color_vec)
            if Xs is None:
                Xs = X
                Xs2 = X**2
            else:
                Xs += X
                Xs2 += X**2

            print('took {:6.1f} seconds'.format(time.time() - tic))
            if i % 5 == 0 or n_iter < 5:
                std = np.sqrt((Xs2 - Xs**2 / i) / (i - 1))
                config.save_features(Xs / i, i, skip=skip,
                                     test=True if run == 'test' else False)
                config.save_std(std, i, skip=skip,
                               test=True if run == 'test' else False)
                print('saved {} iterations'.format(i))


if __name__ == '__main__':
    transform()
