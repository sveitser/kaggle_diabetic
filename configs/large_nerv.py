from layers import *

from lasagne import layers 

from config import Config

cnf = {
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_res',
    'batch_size': 112,
}

layers = [
    (layers.InputLayer, {'shape': (cnf['batch_size'], 3, cnf['w'], cnf['h'])}),
    (cuda_convnet.ShuffleBC01ToC01BLayer, {}),
    (Conv2DLayer, conv_params(32, stride=(2, 2))),
    (Conv2DLayer, conv_params(32)),
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
    (MaxPool2DLayer, pool_params(stride=(1, 1))),
    (cuda_convnet.ShuffleC01BToBC01Layer, {}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 2048}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 2048}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
