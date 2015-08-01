from layers import *

from config import Config

cnf = {
    'name': 'micro_4',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 48,
    'batch_size_test': 8,
    'balance_ratio': 0.9,
    'min_balance': 0.02,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(24, filter_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(32)),
    #Conv2DLayer, conv_params(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(64, stride=(2, 2))),
    (Conv2DLayer, conv_params(64)),
    (Conv2DLayer, conv_params(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(512, filter_size=(5, 5))),
    (Conv2DLayer, conv_params(512, filter_size=(5, 5))),
    (RMSPoolLayer, pool_params(pool_size=(6, 6), stride=((6, 6)))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
