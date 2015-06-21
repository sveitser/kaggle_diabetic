from itertools import product

import click
import numpy as np
import matplotlib.pyplot as plt
from nolearn.lasagne import visualize

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import util
import nn
import augment
from definitions import *

def plot_conv_bias(layer, figsize=(12, 12)):
    """Plot the weights of a specific layer. Only really makes sense
    with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    b = layer.b.get_value()
    shape = b.shape
    print(np.max(b), np.min(b))
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows
    for feature_map in range(shape[0]):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(b[feature_map], cmap='jet',
                              interpolation='nearest')


@click.command()
@click.option('--n', default=0, type=int)
@click.option('--cnf', default='config/micro.py')
@click.option('--weights', is_flag=True, default=False)
def plot(n, cnf, weights):

    model = util.load_module(cnf).model

    files = util.get_image_files(model.get('train_dir'))
    net = nn.create_net(model)
    net.load_params_from(model.weights_file)

    if weights:
        plot_weights(net, model)
    else:
        fname = files[n]
        patient = util.get_names(files)[n]

        x = augment.load(fname, **model.cnf)
        x = x[np.newaxis, ...]
        print(x.shape)

        for name, layer in net.layers_.items():
            if 'conv' in name:
                print("plotting layer {}".format(name))
                visualize.plot_conv_activity(layer, x, figsize=(32, 18)),
                plt.savefig('fig/{}_{}_{}.png'.format(model.cnf['name'],
                                                      patient, name))


def plot_weights(net, model):
    for name, layer in net.layers_.items():
        if 'conv' in name:
            print("plotting layer {}".format(name))
            visualize.plot_conv_weights(layer)
            plt.savefig('fig/weights_{}_{}.png'.format(model.cnf['name'],
                                                       name))
            plot_conv_bias(layer)
            plt.savefig('fig/bias_{}_{}.png'.format(model.cnf['name'],
                                                       name))

        



if __name__ == '__main__':
    plot()

