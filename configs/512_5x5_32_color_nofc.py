from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 48,
    'batch_size_test': 16,
    'balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    #'balance_weights':  np.array([1, 20, 4, 40, 40], dtype=float),
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'balance_ratio': 0.9,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.4, 1.4),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'sigma': 0.1,
    'schedule': {
        0: 0.001,
        100: 0.0001,
        150: 0.00001,
        200: 'stop',
    },
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(32, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, conv_params(32, filter_size=(5, 5))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(64, stride=(2, 2))),
    (Conv2DLayer, conv_params(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(512)),
    #(Conv2DLayer, conv_params(384)),
    #(Conv2DLayer, conv_params(384)),
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1}),
]


config = Config(layers=layers, cnf=cnf)
