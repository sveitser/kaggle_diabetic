from __future__ import division
import time

import click
import numpy as np

import nn
import data
import tta
import util

@click.command()
@click.option('--cnf', default='config/c_128_4x4_32.py',
              help="Path or name of configuration module.")
@click.option('--n_iter', default=1,
              help="Iterations for test time averaging.")
@click.option('--skip', default=0,
              help="Number of test time averaging iterations to skip.")
@click.option('--test', is_flag=True, default=False)
@click.option('--train', is_flag=True, default=False)
@click.option('--weights_from', default=None,
              help='Path to initial weights file.', type=str)
def transform(cnf, n_iter, skip, test, train, weights_from):

    config = util.load_module(cnf).config

    runs = {}
    if train:
        runs['train'] = config.get('train_dir')
    if test:
        runs['test'] = config.get('test_dir')

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

        print("transforming {}".format(directory))
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

            print('took {:6.1f}s'.format(time.time() - tic))
            if i % 5 == 0 or n_iter < 5:
                std = np.sqrt((Xs2 - Xs**2 / i) / (i - 1))
                config.save_transform(Xs / i, i, skip=skip,
                                     test=True if run == 'test' else False)
                config.save_std(std, i, skip=skip,
                               test=True if run == 'test' else False)
                print('saved {} iterations'.format(i))


if __name__ == '__main__':
    transform()
