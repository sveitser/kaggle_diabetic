import sys

import click
import numpy as np

from sklearn.utils import shuffle
from sklearn import cross_validation

from quadratic_weighted_kappa import quadratic_weighted_kappa

from definitions import *
import iterator, util
from nn import create_net


class Log(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self, *args, **kwargs):
        self.terminal.flush()
        self.log.flush()


@click.command()
@click.option('--cnf', default='config/best.py',
              help='Path or name of configuration module.')
@click.option('--weights_from', default=None,
              help='Path to initial weights file.')
def main(cnf, weights_from):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    files = util.get_image_files(config.get('train_dir', TRAIN_DIR))

    names = util.get_names(files)
    y = util.get_labels(names).astype(np.float32)

    f_train, f_test, y_train, y_test = util.split(files, y)

    net = create_net(config)

    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")

    print("fitting ...")
    net.fit(files, y)

    print("making predictions on validation set")
    y_pred = np.round(net.predict(f_test)).astype(int)

    print("ConvNet quadratic weighted kappa {}".format(
        quadratic_weighted_kappa(y_test, y_pred)))


if __name__ == '__main__':
    main()

