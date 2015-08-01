from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_small',
    'test_dir': 'data/test_small',
    'batch_size_train': 128,
    'batch_size_test': 16,
    'mean': [108.64628601, 75.86886597, 54.34005737],
    'std': [70.53946096, 51.71475228, 43.03428563],
    #'learning_rate': 0.001,
    'patience': 40,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 1.0,
    #'balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'balance_weights': np.array(CLASS_WEIGHTS),
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
    'color': True,
    'sigma': 0.5,
    'schedule': {
        0: 0.003,
        150: 0.0003,
        201: 'stop',
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
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
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
    #(MaxPool2DLayer, pool_params()),
    #(Conv2DLayer, cp(16 * n)),
    #(Conv2DLayer, cp(16 * n)),
    (RMSPoolLayer, pool_params(stride=(3, 3))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dp(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dp(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                  'nonlinearity': rectify if REGRESSION else softmax}),
]

config = Config(layers=layers, cnf=cnf)
