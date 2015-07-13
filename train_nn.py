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
@click.option('--retrain_until', default=None, type=float,
              help='Retrain until training loss reaches threshold.')
def main(cnf, weights_from, retrain_until):

    model = util.load_module(cnf).model

    if weights_from is None:
        weights_from = model.weights_file
    else:
        weights_from = str(weights_from)

    files = util.get_image_files(model.get('train_dir', TRAIN_DIR))

    names = util.get_names(files)
    y = util.get_labels(names).astype(np.float32)

    from collections import Counter
    print(Counter(y))

    #if retrain_until is None:
    f_train, f_test, y_train, y_test = util.split(files, y)
    #else:
    #    f_train, f_test = files, y

    print(len(f_train))
    print(len(f_test))
    #f_train, f_test, y_train, y_test = util.split(l_files, y2)

    # add load 50% pseudo label images
    #test_files, _ = util.split(util.get_image_files(TEST_DIR), 
    #                           test_size=len(f_train) / 2)
    #test_names = util.get_names(test_files)
    #pseudo_labels = util.get_labels(test_names,
    #                                PSEUDO_LABEL_FILE).astype(np.float32)

    #f_train = np.hstack([f_train, test_files])
    #y_train = np.hstack([y_train, pseudo_labels])

    net = create_net(model, retrain_until=retrain_until)

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
    #try:
    #    main()
    #finally:
    #    iterator.delete_shared_array()
    main()

