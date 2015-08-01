import click
import numpy as np
from sklearn.metrics import confusion_matrix

import data
import util
import nn

@click.command()
@click.option('--cnf', default='config/c_128_4x4_32.py',
              help="Path or name of configuration module.")
@click.option('--weights', default=None,
              help="Path to weights file.", type=str)
def fit(cnf, weights):

    # load params doesn't work with unicode
    if weights is not None:
        weights = str(weights)

    config = util.load_module(cnf).config

    files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(files)
    labels = data.get_labels(names)

    _, f_test, _, y_test = data.split(files, labels)

    net = nn.create_net(config)
    if weights is None:
        net.load_params_from(config.weights_file)
    else:
        net.load_params_from(weights)

    y_pred = np.round(np.clip(net.predict(f_test).flatten(), 
                              min(labels), max(labels))).astype(int)

    print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
    print("kappa {}".format(util.kappa(y_test, y_pred)))

if __name__ == '__main__':
    fit()
