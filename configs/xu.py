from layers import *

from lasagne import layers 

from config import Config

cnf = {
    'name': 'xu',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size_train': 64,
    'batch_size_test': 32,
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3),
    }
    args.update(kwargs)
    return conv_params(**args)

layers = [
    (InputLayer, {'shape': (cnf['batch_size_train'], C, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(24, filter_size=(5, 5), stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(64)),
    (Conv2DLayer, cp(64)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (Conv2DLayer, cp(128)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(256)),
    (Conv2DLayer, cp(256)),
    (RMSPoolLayer, pool_params(stride=(2, 2))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
