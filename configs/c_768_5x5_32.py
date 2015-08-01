from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 672,
    'h': 672,
    'train_dir': 'data/train_large',
    'test_dir': 'data/test_large',
    'batch_size_train': 24,
    'batch_size_test': 8,
    'balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    #'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.975,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0005,
    'sigma': 0.5,
    'schedule': {
        0: 0.0003,
        70: 0.00003,
        100: 'stop',
    },
}

nonlinearity = leaky_rectify

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3),
        'nonlinearity': nonlinearity,
    }
    args.update(kwargs)
    return conv_params(**args)

def dp(num_units, *args, **kwargs):
    args = {
        'num_units': num_units,
        'nonlinearity': nonlinearity,
    }
    args.update(kwargs)
    return dense_params(**args)

n = 32

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(n, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, cp(n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(2 * n, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, cp(2 * n)),
    (Conv2DLayer, cp(2 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(4 * n)),
    (Conv2DLayer, cp(4 * n)),
    (Conv2DLayer, cp(4 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(8 * n)),
    (Conv2DLayer, cp(8 * n)),
    (Conv2DLayer, cp(8 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(16 * n)),
    (Conv2DLayer, cp(16 * n)),
    (RMSPoolLayer, pool_params(pool_size=(5, 5), stride=(4, 4))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dp(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dp(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
