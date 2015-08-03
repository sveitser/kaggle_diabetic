"""Conv Nets training script."""
import click
import numpy as np

import data
import util
from nn import create_net


@click.command()
@click.option('--cnf', default='config/c_128_4x4_tiny.py',
              help='Path or name of configuration module.')
@click.option('--weights_from', default=None,
              help='Path to initial weights file.')
def main(cnf, weights_from):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(files)
    labels = data.get_labels(names).astype(np.float32)

    net = create_net(config)

    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")

    print("fitting ...")
    net.fit(files, labels)

if __name__ == '__main__':
    main()

