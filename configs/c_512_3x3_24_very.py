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
    #'balance_weights':  np.array([1, 2.5, 1.8, 3.5, 4], dtype=float),
    'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.9,
    'final_balance_weights':  np.array([1, 2.5, 1.8, 3.5, 4], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.2, 1.2),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'sigma': 0.1,
    'schedule': {
        0: 0.001,
        150: 0.0001,
        200: 0.00001,
        250: 'stop',
    },
    'weight_decay': 0.0005,
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'nonlinearity': very_leaky_rectify,
    }
    args.update(kwargs)
    return conv_params(**args)

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(24, filter_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(24)),
    #Conv2DLayer, conv_params(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(48, stride=(2, 2))),
    (Conv2DLayer, conv_params(48)),
    (Conv2DLayer, conv_params(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(384)),
    (Conv2DLayer, conv_params(384)),
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1024, 'nonlinearity': very_leaky_rectify}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1024, 'nonlinearity': very_leaky_rectify}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
