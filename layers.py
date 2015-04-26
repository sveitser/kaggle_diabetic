import lasagne
from lasagne import layers, init
from lasagne.nonlinearities import softmax, rectify, leaky_rectify

from definitions import *

# import conv and pool layers
# try CuDNN / cuda convnet / CPU in order
try:
    import lasagne.layers.dnn
    Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
    MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer 
    print("using CUDNN backend")
except ImportError:
    print("failed to load CUDNN backend")
    try:
        import lasagne.layers.cuda_convnet
        Conv2DLayer = lasagne.layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = lasagne.layers.cuda_convnet.MaxPool2DCCLayer
        print("using CUDAConvNet backend")
    except ImportError as exc:
        print("failed to load CUDAConvNet backend")
        Conv2DLayer = lasagne.layers.conv.Conv2DLayer
        MaxPool2DLayer = lasagne.layers.pool.MaxPool2DLayer
        print("using CPU backend")


def conv_params(num_filters, filter_size=3, border_mode='same',
         nonlinearity=leaky_rectify, W=init.GlorotUniform(),
         b=init.Constant(0.01), **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size, 'border_mode': border_mode,
        'nonlinearity': nonlinearity, 'W': W, 'b': b
    }
    args.update(kwargs)
    return args


def pool_params(pool_size=3, stride=(2, 2), **kwargs):
    args = {'pool_size': pool_size, 'stride': stride}
    args.update(kwargs)
    return args


def get_nn_layers():
    nn_layers = [
        (layers.InputLayer, {'shape': (None, C, W, H)}),
        (Conv2DLayer, conv_params(48, stride=(2, 2))),
        (MaxPool2DLayer, pool_params()),
        (layers.DropoutLayer, {'p': 0.2}),
        (Conv2DLayer, conv_params(128, stride=(2, 2))),
        (MaxPool2DLayer, pool_params()),
        (layers.DropoutLayer, {'p': 0.2}),
        (Conv2DLayer, conv_params(256)),
        (Conv2DLayer, conv_params(256)),
        (Conv2DLayer, conv_params(256)),
        (MaxPool2DLayer, pool_params()),
        (layers.DropoutLayer, {'p': 0.2}),
        (Conv2DLayer, conv_params(384)),
        (MaxPool2DLayer, pool_params(stride=(1, 1))),
        (layers.DropoutLayer, {}),
        (layers.DenseLayer, {'num_units': 2048}),
        (layers.FeaturePoolLayer, {'pool_size': 2}),
        (layers.DropoutLayer, {}),
        (layers.DenseLayer, {'num_units': 2048}),
        (layers.FeaturePoolLayer, {'pool_size': 2}),
        (layers.DenseLayer, {'num_units': 1}),
    ]
    return nn_layers
