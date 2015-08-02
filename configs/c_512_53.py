from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 48,
    'batch_size_test': 8,
    'balance_weights':  np.array([1, 20, 4, 40, 40], dtype=float),
    'balance_ratio': 0.95,
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
        0: 0.003,
        150: 0.0003,
        180: 0.00003,
        200: 'stop',
    },
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3),
        'nonlinearity': very_leaky_rectify,
    }
    args.update(kwargs)
    return conv_params(**args)


layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(32, filter_size=(5, 5), stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(64, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, cp(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(512)),
    (RMSPoolLayer, pool_params(pool_size=(3, 3), stride=(2, 2))), # pad to get even x/y
    #(Conv2DLayer, cp(512)),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
