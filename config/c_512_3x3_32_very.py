from layers import *

from model import Model

cnf = {
    'name': __name__.split('.')[-1],
    'w': 472,
    'h': 472,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 64,
    'batch_size_test': 16,
    'mean': [108.64628601, 75.86886597, 54.34005737],
    'std': [70.53946096, 51.71475228, 43.03428563],
    #'learning_rate': 0.001,
    'patience': 40,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 1.0,
    #'balance_weights':  np.array([1, 2, 2, 3, 3.5], dtype=float),
    'balance_weights': np.array(CLASS_WEIGHTS),
    'balance_ratio': 0.96,
    'final_balance_weights':  np.array([1, 2.5, 2, 3, 3.5], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.2, 1.2),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0005,
    'color': True,
    'sigma': 0.1,
    'update': 'adam',
    'schedule': {
        0: 0.0002,
        150: 0.00002,
        200: 0.000002,
        250: 'stop',
    },
}

n = 32

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(n, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, conv_params(n)),
    #Conv2DLayer, conv_params(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(2 * n, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, conv_params(2 * n)),
    (Conv2DLayer, conv_params(2 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(4 * n)),
    (Conv2DLayer, conv_params(4 * n)),
    (Conv2DLayer, conv_params(4 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(8 * n)),
    (Conv2DLayer, conv_params(8 * n)),
    (Conv2DLayer, conv_params(8 * n)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(16 * n)),
    (Conv2DLayer, conv_params(16 * n)),
    #(Conv2DLayer, conv_params(16 * n)),
    #(RMSPoolLayer, pool_params(pool_size=(3, 3))), # pad to get even x/y
    (MaxPool2DLayer, pool_params(stride=(3, 3))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                  'nonlinearity': rectify if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
