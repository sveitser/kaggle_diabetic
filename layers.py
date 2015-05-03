import lasagne
from lasagne import layers, init
import lasagne.layers.normalization
from lasagne.layers.noise import GaussianNoiseLayer
from lasagne.nonlinearities import softmax, rectify, leaky_rectify
from lasagne.layers.normalization import LocalResponseNormalization2DLayer

from theano import tensor as T

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
         b=init.Constant(0.01), untie_biases=True, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size, 'border_mode': border_mode,
        'nonlinearity': nonlinearity, 'W': W, 'b': b,
        'untie_biases': untie_biases,
    }
    args.update(kwargs)
    return args


def pool_params(pool_size=3, stride=(2, 2), **kwargs):
    args = {
        'pool_size': pool_size, 
        'stride': stride,
    }
    args.update(kwargs)
    return args

class RGBMixLayer(layers.Layer):
    def __init__(self, incoming, alpha=0.5, **kwargs):
        """
        :parameters:
            - incoming: input layer or shape
            - alpha: see equation above
            - k: see equation above
            - beta: see equation above
            - n: number of adjacent channels to normalize over.
        """
        super(RGBMixLayer, self).__init__(incoming, **kwargs)
        self.alpha = alpha
        raise NotImplementedError

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        pass


