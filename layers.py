import numpy as np

import lasagne
from lasagne.layers import *
from lasagne import init, layers
import lasagne.layers.normalization
from lasagne.layers.noise import GaussianNoiseLayer
from lasagne.nonlinearities import *
from lasagne.layers.normalization import LocalResponseNormalization2DLayer

from theano import tensor as T
from theano.sandbox.cuda import dnn


# import conv and pool layers
# try CuDNN / cuda convnet / CPU in order
try:
    import lasagne.layers.dnn
    Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
    MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer 
    Pool2DLayer = lasagne.layers.dnn.Pool2DDNNLayer
    print("using CUDNN backend")
except ImportError:
    print("failed to load CUDNN backend")
    try:
        import lasagne.layers.cuda_convnet
        Conv2DLayer = lasagne.layers.cuda_convnet.Conv2DCCLayer
        Pool2DLayer = lasagne.layers.cuda_convnet.Pool2DLayer
        MaxPool2DLayer = lasagne.layers.cuda_convnet.MaxPool2DCCLayer
        print("using CUDAConvNet backend")
    except ImportError as exc:
        print("failed to load CUDAConvNet backend")
        Conv2DLayer = lasagne.layers.conv.Conv2DLayer
        MaxPool2DLayer = lasagne.layers.pool.MaxPool2DLayer
        Pool2DLayer = MaxPool2DLayer
        print("using CPU backend")


# try to use theano meta optimizations
C2DNoStrideLayer = lasagne.layers.conv.Conv2DLayer

# nervana kernels
#import nervana_theano.layers
#from lasagne.layers import cuda_convnet
#Conv2DLayer = nervana_theano.layers.NervanaConvLayer
#import lasagne.layers.cuda_convnet
#MaxPool2DLayer = lasagne.layers.cuda_convnet.MaxPool2DCCLayer

class SharedRelu(LeakyRectify):
    def __call__(self, x):
        # The following is faster than T.maximum(leakiness * x, x),
        # and it works with nonsymbolic inputs as well. Also see:
        # https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81765117
        f1 = 0.5 * (1 + self.leakiness)
        f2 = 0.5 * (1 - self.leakiness)
        return f1 * x + f2 * abs(x)


def conv_params(num_filters, filter_size=(3, 3), border_mode='same',
         nonlinearity=leaky_rectify, W=init.Orthogonal(gain=1.0),
         b=init.Constant(0.05), untie_biases=True, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size, 
        'border_mode': border_mode,
        'nonlinearity': nonlinearity, 
        'W': W, 
        'b': b,
        'untie_biases': untie_biases,
        #'dimshuffle': False, # set False for nervana conv layer
    }
    args.update(kwargs)
    return args


def pool_params(pool_size=3, stride=(2, 2), **kwargs):
    args = {
        'pool_size': pool_size, 
        'stride': stride,
        #'dimshuffle': False, # set False for convnet pool layer
    }
    args.update(kwargs)
    return args

def dense_params(num_units, nonlinearity=leaky_rectify, **kwargs):
    args = {
        'num_units': num_units, 
        'nonlinearity': nonlinearity,
        'W': init.Orthogonal(1.0),
        'b': init.Constant(0.05),
    }
    args.update(kwargs)
    return args

# from https://github.com/benanne/kaggle-ndsb/blob/master/tmp_dnn.py
class RMSPoolLayer(Pool2DLayer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0), 
                 epsilon=1e-12, **kwargs):
        super(RMSPoolLayer, self).__init__(incoming, pool_size,  stride,
                                           pad, **kwargs)
        self.epsilon = epsilon
        #del self.mode # TODO check if this is needed

    def get_output_for(self, input, *args, **kwargs):
        out = dnn.dnn_pool(T.sqr(input), self.pool_size, self.stride, 
                           'average')
        return T.sqrt(out + self.epsilon)

class AveragePoolLayer(Pool2DLayer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0), **kwargs):
        super(AveragePoolLayer, self).__init__(incoming, pool_size,  stride,
                                               pad, **kwargs)
        del self.mode

    def get_output_for(self, input, *args, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride, 'average')
