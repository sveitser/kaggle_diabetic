from layers import *

from config import Config

cnf = {
    'name': '512_5x5_24',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 64,
    'batch_size_test': 16,
    'balance_weights':  np.array([1, 20, 4, 40, 40], dtype=float),
    'balance_ratio': 0.8,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
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
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(24, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, conv_params(24, filter_size=(5, 5))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(48, filter_size=(4, 4), stride=(2, 2))),
    (Conv2DLayer, conv_params(48, filter_size=(4, 4))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(384)),
    (Conv2DLayer, conv_params(384)),
    (Conv2DLayer, conv_params(384)),
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 512}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 512}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]


config = Config(layers=layers, cnf=cnf)
