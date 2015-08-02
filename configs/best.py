from layers import *

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (GaussianNoiseLayer, {'sigma': 0.05}),
    (Conv2DLayer, conv_params(48, stride=(2, 2), filter_size=(3, 3))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(96, stride=(2, 2))),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (Conv2DLayer, conv_params(192)),
    (MaxPool2DLayer, pool_params()),
    (layers.DropoutLayer, {'p': 0.2}),
    (Conv2DLayer, conv_params(256)),
    (MaxPool2DLayer, pool_params(stride=(1, 1))),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 2048}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 2048}),
    (layers.FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]


