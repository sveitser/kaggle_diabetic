from layers import *

from config import Config

cnf = {
    'name': '256_3x3_32_feat_only',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size_train': 128,
    'batch_size_test': 32,
    #'balance_weights':  np.array([1, 10.5, 4.8, 29.5, 36.4], dtype=float),
    'balance_weights':  np.array([1, 4, 2.5, 4, 6], dtype=float),
    'balance_ratio': 1.0,
    'final_balance_weights':  np.array([1, 4, 2, 4, 4], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.1, 1.1),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-20, 20),
        'do_flip': True,
        'allow_stretch': True,
    }
}

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(32, filter_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(64, stride=(2, 2))),
    (Conv2DLayer, conv_params(64)),
    (Conv2DLayer, conv_params(64)),
    (Conv2DLayer, conv_params(64)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (Conv2DLayer, conv_params(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    (Conv2DLayer, conv_params(256)),
    #(MaxPool2DLayer, pool_params()),
    #(Conv2DLayer, conv_params(512)),
    #(Conv2DLayer, conv_params(256)),
    #(Conv2DLayer, conv_params(256)),
    (RMSPoolLayer, pool_params(stride=(2, 2))), # pad to get even x/y
    (DropoutLayer, {'p': 0.5}),
    #(DenseLayer, {'num_units': 512, 'nonlinearity': leaky_rectify}),
    #(FeaturePoolLayer, {'pool_size': 2}),
    #(DenseLayer, {'num_units': 512, 'nonlinearity': leaky_rectify}),
    #(FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
