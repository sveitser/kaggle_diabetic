from layers import *

from config import Config

cnf = {
    'name': 'micro_5',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 40,
    'batch_size_test': 8,
    'balance_ratio': 0.8,
    'min_balance': 0.02,
}

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
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
    (Conv2DLayer, conv_params(384, filter_size=(5, 5))),
    (Conv2DLayer, conv_params(384, filter_size=(5, 5))),
    (RMSPoolLayer, pool_params(pool_size=(6, 6), stride=((6, 6)))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    #(RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    #(DropoutLayer, {'p': 0.5}),
    #(DenseLayer, {'num_units': 2048}),
    #(FeaturePoolLayer, {'pool_size': 2}),
    #(DropoutLayer, {'p': 0.5}),
    #(DenseLayer, {'num_units': 2048}),
    #(FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
