from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 448,
    'h': 448,
    #'train_dir': '/data/kaggle/diabetic/train_medium', #TODO FIX
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 40,
    'batch_size_test': 8,
    #'balance_weights':  np.array([1, 2.5, 1.8, 3.5, 4], dtype=float),
    'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.975,
    'final_balance_weights':  np.array([1, 2.5, 1.8, 3.5, 4], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.4, 1.4),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0005,
    'sigma': 0.2,
    'schedule': {
        0: 0.002,
        150: 0.0002,
        200: 0.00002,
        250: 'stop',
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
    (InputLayer, {'shape': (cnf['batch_size_train'], 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, cp(24, stride=(2, 2))),
    (Conv2DLayer, cp(24, border_mode=None, pad=2)),
    #Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48, stride=(2, 2))),
    (Conv2DLayer, cp(48, border_mode=None, pad=2)),
    (Conv2DLayer, cp(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(96, border_mode=None, pad=2)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96, border_mode=None, pad=2)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192, border_mode=None, pad=2)),
    (Conv2DLayer, cp(192)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(384, border_mode=None, pad=2)),
    #(Conv2DLayer, cp(384)),
    #(Conv2DLayer, cp(384, border_mode=None, pad=2)),
    (RMSPoolLayer, pool_params(pool_size=(3, 3), stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
