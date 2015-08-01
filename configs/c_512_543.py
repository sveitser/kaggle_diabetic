from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 40,
    'batch_size_test': 16,
    'balance_weights': np.array([1.360, 14.37, 6.637, 40.23, 49.61]),
    'balance_ratio': 0.96,
    'final_balance_weights':  np.array([1, 2.5, 1.8, 3.5, 4.5], dtype=float),
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
        0: 0.003,
        150: 0.0003,
        200: 0.00003,
        220: 'stop',
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
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(16, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(64, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, cp(64)),
    (Conv2DLayer, cp(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    #(MaxPool2DLayer, pool_params(pool_size=(4, 4), stride=(3, 3))),
    #(Conv2DLayer, cp(512)),
    #(Conv2DLayer, cp(512)),
    (RMSPoolLayer, pool_params(pool_size=(3, 3), stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(512)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(512)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                  'nonlinearity': None if REGRESSION else softmax}),
]

config = Config(layers=layers, cnf=cnf)
