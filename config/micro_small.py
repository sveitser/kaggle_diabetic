from layers import *

from model import Model

cnf = {
    'name': 'micro',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'batch_size': 32,
    #'mean': [112.26],
    #'std': [26.63],
    'mean': [ 108.73683167, 75.54026794,  53.80962753],
    'std': [ 70.44262987, 51.35997035, 42.51656026],
    'learning_rate': 0.005,
    'regression': True,
    #'n_classes': 3,
    'rotate': True,
    'balance': 0.1,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(24, stride=(2, 2))),
    (Conv2DLayer, conv_params(24)),
    #Conv2DLayer, conv_params(32)),
    (RMSPoolLayer, pool_params()),
    (Conv2DLayer, conv_params(48, stride=(2, 2))),
    (Conv2DLayer, conv_params(48)),
    (Conv2DLayer, conv_params(48)),
    (RMSPoolLayer, pool_params()),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (Conv2DLayer, conv_params(96)),
    (RMSPoolLayer, pool_params()),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (RMSPoolLayer, pool_params(stride=(2, 2))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                         'nonlinearity': rectify if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
