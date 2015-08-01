from layers import *

from config import Config

cnf = {
    'name': '512_3x3_24_double_triple.py',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 48,
    'batch_size_test': 8,
    'balance_weights':  np.array([1, 10.5, 4.8, 29.5, 36.4], dtype=float),
    'final_balance_weights':  np.array([1, 3.8, 2.0, 4.0, 5.5], dtype=float),
    'balance_ratio': 0.0,
    'aug_params': {
        'zoom_range': (1 / 1.1, 1.1),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    }
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(24, filter_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(24)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(48, stride=(2, 2))),
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
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
