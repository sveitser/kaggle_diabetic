from layers import *

from model import Model

cnf = {
    'name': '256_3x3_16',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size_train': 128,
    'batch_size_test': 32,
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
    'balance_weights':  np.array([1, 10.5, 4.8, 29.5, 36.4], dtype=float),
    'final_balance_weights':  np.array([1, 2, 1.5, 3.5, 4.5], dtype=float),
    'balance_ratio': 0.8,
    'aug_params': {
        'zoom_range': (1 / 1.1, 1.1),
        'rotation_range': (-15, 15),
        'shear_range': (0, 0),
        'translation_range': (-20, 20),
        'do_flip': False,
        'allow_stretch': True,
    }
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(16, filter_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(16)),
    (Conv2DLayer, conv_params(16)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(32, stride=(2, 2))),
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
    #(MaxPool2DLayer, pool_params()),
    #(Conv2DLayer, conv_params(512)),
    #(Conv2DLayer, conv_params(256)),
    #(Conv2DLayer, conv_params(256)),
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 1024}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1024}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': N_TARGETS if REGRESSION else N_CLASSES,
                         'nonlinearity': rectify if REGRESSION else softmax}),
]

model = Model(layers=layers, cnf=cnf)
