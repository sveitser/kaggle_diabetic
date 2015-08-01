from layers import *

from lasagne import layers 

from config import Config

cnf = {
    'name': 'large_ordinal',
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'test_dir': 'data/test_res',
    'batch_size': 128,
    'ordinal': True,
}

layers = [
    (InputLayer, {'shape': (cnf['batch_size'], 3, cnf['w'], cnf['h'])}),
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
    (RMSPoolLayer, pool_params(stride=(1, 1))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 2048}),
    (FeaturePoolLayer, {'pool_size': 2}),
]

config = Config(layers=layers, cnf=cnf)
