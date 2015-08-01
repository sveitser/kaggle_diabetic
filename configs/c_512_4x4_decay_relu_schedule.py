from layers import *

from config import Config

cnf = {
    'name': __name__.split('.')[-1],
    'w': 472,
    'h': 472,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 64,
    'batch_size_test': 8,
    #'balance_weights':  np.array([1, 2, 2, 3, 3.5], dtype=float),
    'balance_weights': np.array(CLASS_WEIGHTS),
    'balance_ratio': 0.95,
    'final_balance_weights':  np.array([1, 2, 2, 3, 3.5], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.2, 1.2),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0005,
    'sigma': 0.1,
    #'update': 'adam',
    'schedule': {
        0: 0.002,
        150: 0.0002,
        200: 0.00002,
        221: 'stop',
    },
    'slope': theano.shared(np.cast['float32'](0.35)),
    'slope_decay': 0.975,
    'slope_schedule': {
        50: 0.15,
        100: 0.05,
    },
}

#nonlinearity = SharedRelu(cnf['slope'])
nonlinearity = LeakyRectify(cnf['slope'])

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (4, 4),
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
    (Conv2DLayer, cp(n, stride=(2, 2))),
    (Conv2DLayer, cp(n, border_mode=None, pad=2)),
    #Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(2 * n, stride=(2, 2))),
    (Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
    (Conv2DLayer, cp(2 * n)),
    #(Conv2DLayer, cp(2 * n, border_mode=None, pad=2)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
    (Conv2DLayer, cp(4 * n)),
    (Conv2DLayer, cp(4 * n, border_mode=None, pad=2)),
    #(Conv2DLayer, cp(4 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(8 * n)),
    (Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
    (Conv2DLayer, cp(8 * n)),
    #(Conv2DLayer, cp(8 * n, border_mode=None, pad=2)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(16 * n)),
    #(Conv2DLayer, cp(384, border_mode=None, pad=2)),
    (RMSPoolLayer, pool_params()),
    #(MaxPool2DLayer, pool_params()),
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
