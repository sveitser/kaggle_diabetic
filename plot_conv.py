import click
import numpy as np
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize

import util
import nn
from definitions import *


@click.command()
@click.option('--n', default=0, type=int)
@click.option('--cnf', default='config/best.py')
def plot(n, cnf):

    cnf = util.load_module(cnf)

    mean = util.get_mean()
    files = util.get_image_files(TRAIN_DIR)
    net = nn.create_net(mean, cnf.layers)
    net.load_weights_from(WEIGHTS)

    filename = files[n]

    patient = util.get_names([filename])[0]

    x = (util.load_image(filename) - mean) / MAX_PIXEL_VALUE
    x = x[np.newaxis, ...]
    for name, layer in net.layers_.items():
        if 'conv' in name:
            print("plotting layer {}".format(name))
            visualize.plot_conv_activity(layer, x)
            plt.savefig('fig/{}_{}.png'.format(patient, name))


if __name__ == '__main__':
    plot()

