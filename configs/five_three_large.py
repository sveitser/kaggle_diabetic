from layers import *

from lasagne import layers 

from config import Config

cnf = {
    'name': 'five_three_large',
    'w': 448,
    'h': 448,
    'train_dir': 'data/train_medium',
    'test_dir': 'data/test_medium',
    'batch_size_train': 64,
    'batch_size_test': 16,
}

def cp(num_filters, *args, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': (3, 3)
    }
    args.update(kwargs)
    return conv_params(**args)

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(16, filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, cp(24, stride=(2, 2))),
    #Conv2DLayer, cp(32)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (Conv2DLayer, cp(48)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (Conv2DLayer, cp(96)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192)),
    (Conv2DLayer, cp(192)),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, cp(384)),
    (RMSPoolLayer, pool_params()),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
