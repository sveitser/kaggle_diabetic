from layers import *

from config import Config

cnf = {
    'w': 256,
    'h': 256,
    'train_dir': 'data/train_medium',
    'batch_size': 128,
}


layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(16, filter_size=(3, 3), stride=(2, 2))),
    (MaxPool2DLayer, pool_params(pool_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(32, stride=(2, 2))),
    (MaxPool2DLayer, pool_params(pool_size=(3, 3), stride=(2, 2))),
    (Conv2DLayer, conv_params(64, stride=(2, 2))),
    (Conv2DLayer, conv_params(128, stride=(1, 1))),
    (MaxPool2DLayer, pool_params(pool_size=(3, 3), stride=(2, 2))),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 1024}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 1024}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
