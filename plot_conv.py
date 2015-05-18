import click
import numpy as np
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize

import util
import nn
import augment
from definitions import *


@click.command()
@click.option('--n', default=0, type=int)
@click.option('--cnf', default='config/micro.py')
def plot(n, cnf):

    model = util.load_module(cnf).model

    files = util.get_image_files(model.get('train_dir'))
    net = nn.create_net(model)
    net.load_params_from(WEIGHTS)

    fname = files[n]
    patient = util.get_names(files)[n]

    x = augment.load(fname, **model.cnf)
    x = x[np.newaxis, ...]
    print(x.shape)

    for name, layer in net.layers_.items():
        if 'conv' in name:
            print("plotting layer {}".format(name))
            visualize.plot_conv_activity(layer, x)
            plt.savefig('fig/{}_{}.png'.format(patient, name))


if __name__ == '__main__':
    plot()

