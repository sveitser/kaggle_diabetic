from layers import *

from model import Model

cnf = {
    'name': 'micro_5',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 40,
    'batch_size_test': 8,
    #'mean': [112.26],
    #'std': [26.63],
    'mean': [ 108.73683167, 75.54026794,  53.80962753],
    'std': [ 70.44262987, 51.35997035, 42.51656026],
    'learning_rate': 0.003,
    'patience': 40,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 1.0,
    'balance_ratio': 0.8,
    'min_balance': 0.02,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
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
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                         'nonlinearity': rectify if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
