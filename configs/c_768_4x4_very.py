from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 672,
    'h': 672,
    'train_dir': 'data/train_large',
    'test_dir': 'data/test_large',
    'batch_size_train': 40,
    'batch_size_test': 8,
    #'balance_weights':  np.array([1, 2.0, 2.0, 2.5, 3.0], dtype=float),
    'balance_weights':  np.array([1, 2.0, 2.0, 2.5, 3.0], dtype=float),
    'balance_ratio': 0.5,
    'final_balance_weights':  np.array([1, 2.0, 2.0, 2.5, 3.0], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.3, 1.3),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-80, 80),
        'do_flip': True,
        'allow_stretch': True,
    },
    'sigma': 0.1,
    'schedule': {
        0: 0.001,
        50: 0.0001,
        100: 0.00001,
        120: 'stop',
        #0: 0.001,
        #100: 0.0001,
        #150: 0.00001,
        #170: 'stop',
    },
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (4, 4),
        'nonlinearity': very_leaky_rectify,
    }
    args.update(kwargs)
    return conv_params(**args)

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(24, stride=(2, 2))),
    (Conv2DLayer, cp(24)),
    #Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48, stride=(2, 2))),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(384)),
    #(Conv2DLayer, cp(384)),
    (RMSPoolLayer, pool_params(pool_size=(3, 3), stride=(3, 3))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
