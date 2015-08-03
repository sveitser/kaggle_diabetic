import lasagne
from lasagne.layers import DenseLayer, InputLayer, FeaturePoolLayer
from lasagne import init, layers
from lasagne.nonlinearities import leaky_rectify

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


def dense_params(num_units, nonlinearity=leaky_rectify, **kwargs):
    args = {
        'num_units': num_units, 
        'nonlinearity': nonlinearity,
        'W': init.Orthogonal(1.0),
        'b': init.Constant(0.05),
    }
    args.update(kwargs)
    return args


class RMSPoolLayer(Pool2DLayer):
    """Use RMS as pooling function.

    from https://github.com/benanne/kaggle-ndsb/blob/master/tmp_dnn.py
    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0), 
                 epsilon=1e-12, **kwargs):
        super(RMSPoolLayer, self).__init__(incoming, pool_size,  stride,
                                           pad, **kwargs)
        self.epsilon = epsilon
        del self.mode

    def get_output_for(self, input, *args, **kwargs):
        out = dnn.dnn_pool(T.sqr(input), self.pool_size, self.stride, 
                           'average')
        return T.sqrt(out + self.epsilon)

