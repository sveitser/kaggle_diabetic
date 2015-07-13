"""Provides neural-network specific Ops.

:note: TODO: factor this out into a neural-network toolbox.

:note: We register all optimization with the gpu tag as we don't
    implement all the intermediate case on the GPU (in particular
    AdvancedSubtensor). So to make sure it run well on the gpu with
    fast_compile, we register them as needed for the GPU. This can be
    revisited later when all the intermediate part are on the GPU.

"""
import logging
import numpy

import theano
from theano import gof
from theano.tensor import basic as tensor
from theano.tensor import subtensor
from theano.tensor import elemwise
from theano.tensor import opt
from theano.compile import optdb
from theano.gof import Apply

from theano.tensor.nnet.sigm import sigmoid, softplus
from theano.gradient import DisconnectedType
from theano.gradient import grad_not_implemented
from theano.tensor.type import values_eq_approx_remove_nan


class CrossentropyCategorical1HotGrad(gof.Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, g_y, coding_dist, true_one_of_n):
        return Apply(self, [g_y, coding_dist, true_one_of_n],
                     [coding_dist.type()])

    def perform(self, node, inp, out):
        g_y, coding_dist, true_one_of_n = inp
        g_coding_strg, = out
        g_coding = numpy.zeros_like(coding_dist)
        for i in xrange(len(g_y)):
            true_dist = numpy.ones_like(g_coding[0, :])
            true_dist[true_one_of_n[i]:] = 0
            #print(true_dist, true_dist.shape)
            #print(coding_dist, coding_dist.shape)
            #print(g_coding, g_coding.shape)
            g_coding[i, :] = - 2.0 * coding_dist[i] \
                    * (true_dist - coding_dist[i]) * (1 - coding_dist[i])
            #g_coding[i, true_one_of_n[i]] = -g_y[i] / coding_dist[i,
            #                                            true_one_of_n[i]]
        g_coding_strg[0] = g_coding

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

crossentropy_categorical_1hot_grad = CrossentropyCategorical1HotGrad()


class CrossentropyCategorical1Hot(gof.Op):

    """Compute the cross entropy between a coding distribution and
    a true distribution of the form [0, 0, ... 0, 1, 0, ..., 0]

    .. math::

        y[i] = - \log(coding_dist[i, one_of_n[i])


    :note: In the case that the coding distribution is the output of a
           softmax, an application of this Op will probably be optimized
           away in favour of one with a C implementation.

    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, coding_dist, true_one_of_n):
        """
        :type coding_dist: dense matrix

        :type true_one_of_n: lvector

        :rtype: dvector
        """
        _coding_dist = tensor.as_tensor_variable(coding_dist)
        _true_one_of_n = tensor.as_tensor_variable(true_one_of_n)
        if _coding_dist.type.ndim != 2:
            raise TypeError('matrix required for argument: coding_dist')
        if _true_one_of_n.type not in (tensor.lvector, tensor.ivector):
            raise TypeError(
                'integer vector required for argument: true_one_of_n'
                '(got type: %s instead of: %s)' % (_true_one_of_n.type,
                                                   tensor.lvector))

        return Apply(self, [_coding_dist, _true_one_of_n],
                [tensor.Tensor(dtype=_coding_dist.dtype,
                               broadcastable=[False])()])

    def perform(self, node, inp, out):
        coding, one_of_n = inp
        y_out, = out
        y = numpy.zeros_like(coding[:, 0])
        for i in xrange(len(y)):
            #y[i] = -numpy.log(coding[i, one_of_n[i]])
            true_dist = numpy.zeros_like(coding[0, :])
            true_dist[one_of_n[i]:] = 1
            y[i] = numpy.sum((coding - true_dist)**2)
        y_out[0] = y

# Enabling this infer_shape method make 2 tests fail:
# theano/tensor/nnet/tests/test_nnet.py:T_CrossentropyCategorical1Hot.
#     {test_softmax_grad_optimizations,test_softmax_grad_optimizations_vector}
# This is caused by the local_fill_to_alloc that call broadcast_like
# that look into the shape feature and return a Rebroadcast instead of an alloc.
# I disable this infer_shape until we fix the optimizations or determine that
# this is not needed anymore and we update the tests.
        # see issue gh-788
#    def infer_shape(self, node, in_shapes):
#        return [(in_shapes[0][0],)]

    def grad(self, inp, grads):
        coding, one_of_n = inp
        g_y, = grads
        return [crossentropy_categorical_1hot_grad(g_y, coding, one_of_n),
                grad_not_implemented(self, 1, one_of_n)]

crossentropy_categorical_1hot = CrossentropyCategorical1Hot()

ordinal_loss = CrossentropyCategorical1Hot()

