from layers import *

layers = [
    (layers.InputLayer, {'shape': (None, 3, W, H)}),
    (Conv2DLayer, conv_params(32, stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.2}),
    (Conv2DLayer, conv_params(48, stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.2}),
    (Conv2DLayer, conv_params(64)),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.2}),
    (Conv2DLayer, conv_params(128)),
    (MaxPool2DLayer, pool_params(stride=(1, 1))),
    (layers.DropoutLayer, {}),
    (layers.DenseLayer, {'num_units': 512}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {}),
    (layers.DenseLayer, {'num_units': 512}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (layers.DenseLayer, {'num_units': 1}),
]
