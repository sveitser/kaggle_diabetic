from layers import *

from model import Model

cnf = {
    'name': 'micro_4',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 48,
    'batch_size_test': 8,
    #'mean': [112.26],
    #'std': [26.63],
    'mean': [ 108.73683167, 75.54026794,  53.80962753],
    'std': [ 70.44262987, 51.35997035, 42.51656026],
    'learning_rate': 0.003,
    'patience': 30,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 1.0,
    'balance_ratio': 0.9,
    'min_balance': 0.02,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
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
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                         'nonlinearity': rectify if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
