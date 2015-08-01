from layers import *

layers = [
    (layers.InputLayer, {'shape': (None, C, W, H)}),
    (Conv2DLayer, conv_params(32, stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.2}),
    (Conv2DLayer, conv_params(64, stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.2}),
    (Conv2DLayer, conv_params(128, stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 1024}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 1024}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]


