import importlib

import click
import numpy as np

from sklearn.utils import shuffle
from sklearn import cross_validation

from quadratic_weighted_kappa import quadratic_weighted_kappa

from definitions import *
import util
from nn import create_net

@click.command()
@click.option('--cnf', default='config/best.py',
              help='Path or name of configuration module.')
def main(cnf):

    config = importlib.import_module(cnf.replace('/', '.').strip('.py'))

    print('loading data...')
    files = util.get_image_files(TRAIN_DIR)
    #l_files = util.get_image_files(TRAIN_DIR, left_only=True)

    names = util.get_names(files)
    y = util.get_labels(names).astype(np.float32)
    #y2 = util.get_labels(names, per_patient=True).astype(np.float32)

    f_train, f_test, y_train, y_test = util.split(files, y)
    #f_train, f_test, y_train, y_test = util.split(l_files, y2)

    # add load 50% pseudo label images
    #test_files, _ = util.split(util.get_image_files(TEST_DIR), 
    #                           test_size=len(f_train) / 2)
    #test_names = util.get_names(test_files)
    #pseudo_labels = util.get_labels(test_names,
    #                                PSEUDO_LABEL_FILE).astype(np.float32)

    #f_train = np.hstack([f_train, test_files])
    #y_train = np.hstack([y_train, pseudo_labels])

    mean = util.get_mean(files)

    net = create_net(mean, config.layers)

    try:
        net.load_weights_from(WEIGHTS)
        print("loaded weights from {}".format(WEIGHTS))
    except Exception:
        print("couldn't load weights starting from scratch")

    print("fitting ...")
    net.fit(f_train, y_train)

    print("saving weights")
    net.save_weights_to(WEIGHTS)

    print("extracting features ...")
    X_train = net.transform(f_train)
    X_test = net.transform(f_test)


    np.save(open(TRAIN_FEATURES, 'wb'), X_train)
    np.save(open(TEST_FEATURES, 'wb'), X_test)
    np.save(open(TRAIN_LABELS, 'wb'), y_train)
    np.save(open(TEST_LABELS, 'wb'), y_test)

    print("making predictions on validation set")
    y_pred = np.round(net.predict(f_test)).astype(int)

    print("ConvNet quadratic weighted kappa {}".format(
        quadratic_weighted_kappa(y_test, y_pred)))

if __name__ == '__main__':
    main()
