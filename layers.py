import settings

import theano
from theano import tensor as T
import lasagne
from lasagne.layers import (DenseLayer, InputLayer, FeaturePoolLayer,
                            DropoutLayer)
from lasagne import init, layers
from lasagne.nonlinearities import leaky_rectify

Conv2DLayer = lasagne.layers.Conv2DLayer
MaxPool2DLayer = lasagne.layers.MaxPool2DLayer
Pool2DLayer = lasagne.layers.Pool2DLayer

GPU = True if theano.config.device.startswith('gpu') else False
CC = False

if GPU:
    if settings.deterministic:
        import lasagne.layers.cuda_convnet
        Conv2DLayer = lasagne.layers.cuda_convnet.Conv2DCCLayer
        MaxPool2DLayer = lasagne.layers.cuda_convnet.MaxPool2DCCLayer
        CC = True
        print("using CUDA-convnet (for determinism)")
    else:
        try:
            import lasagne.layers.dnn
            Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
            MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer
            print("using cuDNN")
        except ImportError:
            print("couldn't load cuDNN layers")
else:
    print("using CPU")

def conv_params(num_filters, filter_size=(3, 3), pad=1,#border_mode='same',
         nonlinearity=leaky_rectify, W=init.Orthogonal(gain=1.0),
         b=init.Constant(0.05), untie_biases=True, **kwargs):
    args = {
        'num_filters': num_filters,
        'filter_size': filter_size,
        #'border_mode': border_mode,
        'pad': pad,
        'nonlinearity': nonlinearity,
        'W': W,
        'b': b,
        'untie_biases': untie_biases,
    }
    if CC:
        args['dimshuffle'] = False
    else:
        args.pop('partial_sum', None)
    args.update(kwargs)
    return args


def pool_params(pool_size=3, stride=(2, 2), **kwargs):
    args = {
        'pool_size': pool_size,
        'stride': stride,
    }
    if CC:
        args['dimshuffle'] = False
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
                 dimshuffle=False, # ignored
                 epsilon=1e-12, **kwargs):
        super(RMSPoolLayer, self).__init__(incoming, pool_size,  stride,
                                           pad, **kwargs)
        self.epsilon = epsilon
        if hasattr(self, 'mode'):
            del self.mode

    def get_output_for(self, input, *args, **kwargs):
        if GPU:
            from theano.sandbox.cuda import dnn
            out = dnn.dnn_pool(T.sqr(input), self.pool_size, self.stride,
                               'average')
        else:
            out = T.signal.downsample.max_pool_2d(
                T.sqr(input), ds=self.pool_size, st=self.stride,
                mode='average_inc_pad')
        return T.sqrt(out + self.epsilon)

