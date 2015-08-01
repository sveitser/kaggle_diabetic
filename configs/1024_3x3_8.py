from layers import *

from config import Config

cnf = {
    'name': '1024_3x3_8',
    'w': 896,
    'h': 896,
    'train_dir': 'data/train_large',
    'test_dir': 'data/test_large',
    'batch_size_train': 32,
    'batch_size_test': 4,
    'balance_weights':  np.array([1, 10.5, 4.8, 29.5, 36.4], dtype=float),
    'final_balance_weights':  np.array([1, 2, 1.5, 2.5, 3], dtype=float),
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
    (Conv2DLayer, conv_params(8, filter_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(8)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(16, stride=(2, 2))),
    (Conv2DLayer, conv_params(16)),
    (Conv2DLayer, conv_params(16)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(32)),
    (Conv2DLayer, conv_params(32)),
    (Conv2DLayer, conv_params(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(64)),
    (Conv2DLayer, conv_params(64)),
    (Conv2DLayer, conv_params(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(256)),
    #(Conv2DLayer, conv_params(256)),
    #(Conv2DLayer, conv_params(256)),
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1024}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1024}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
