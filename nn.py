import Queue
import multiprocessing
from multiprocessing import Pool
import threading

import pandas as pd
import numpy as np

import lasagne
import lasagne.layers
from lasagne import init

from lasagne.updates import nesterov_momentum
from lasagne import updates
from lasagne.objectives import Objective
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano.tensor import Tensor as T

from definitions import *
from quadratic_weighted_kappa import quadratic_weighted_kappa
import util

import augment


def create_net(mean, layers):
    net = NeuralNet(
        layers=layers,
        batch_iterator_train=SingleIterator(batch_size=BATCH_SIZE,
                                               mean=mean,
                                               deterministic=False),
        batch_iterator_test=SingleIterator(batch_size=BATCH_SIZE,
                                              mean=mean,
                                              deterministic=True),
        update=updates.nesterov_momentum,
        update_learning_rate=theano.shared(float32(INITIAL_LEARNING_RATE)),
        update_momentum=theano.shared(float32(INITIAL_MOMENTUM)),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=INITIAL_LEARNING_RATE,
                           stop=0.0001),
            AdjustVariable('update_momentum', start=INITIAL_MOMENTUM,
                            stop=0.999),
            EarlyStopping(loss=CUSTOM_SCORE_NAME),
        ],
        custom_score=(CUSTOM_SCORE_NAME, util.kappa),
        objective=RegularizedObjective,
        use_label_encoder=False,
        eval_size=0.1,
        regression=REGRESSION,
        max_epochs=MAX_ITER,
        verbose=2,
    )
    return net


def float32(k):
    return np.cast['float32'](k)


class RegularizedObjective(Objective):

    def get_loss(self, input=None, target=None, deterministic=False, **kwargs):
        loss = super(RegularizedObjective, self).get_loss(
            input=input, target=target, deterministic=deterministic, **kwargs)
        if not deterministic:
            return loss \
                + 0.0005 * lasagne.regularization.l2(self.input_layer)
        else:
            return loss


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class QueueIterator(BatchIterator):
    """BatchIterator with seperate thread to do the image reading."""
    def __iter__(self):
        queue = Queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for Xb, yb in super(QueueIterator, self).__iter__():
                queue.put((np.array(Xb), np.array(yb)))
            queue.put(end_marker)

        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not end_marker:
            yield item
            queue.task_done()
            item = queue.get()

class SingleIterator(QueueIterator):
    """BatchIterator with flipping and mean subtraction.

    Parameters
    ----------
    mean: np.array, dtype=np.float32
        with shape Channels x Width x Height

    batch_size: int
    """
    def __init__(self, mean, deterministic=False, *args, **kwargs):
        self.mean = mean
        self.deterministic = deterministic
        super(SingleIterator, self).__init__(*args, **kwargs)

    def __iter__(self):

        # make a copy of the original samples
        if not hasattr(self, 'X_orig'):
            print("making a copy of original samples")
            self.X_orig = self.X.copy()
            if self.y is not None:
                self.y_orig = self.y.copy()

        # balance classes in dataset
        if self.y is not None and not self.deterministic:
            n = len(self.y)
            indices = util.balance_shuffle_indices(self.y_orig, 
                                                   random_state=None)
            self.X = self.X_orig[indices[:n]]
            self.y = self.y_orig[indices[:n]]

        return super(SingleIterator, self).__iter__()

    def transform(self, Xb, yb):
        files, labels = super(SingleIterator, self).transform(Xb, yb)
        bs = len(files)

        Xb = (util.load_image(files) - self.mean) / MAX_PIXEL_VALUE

        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]

        #rotations = np.random.randint(4, size=bs)
        #for i, r in enumerate(rotations):
        #    Xb[i] = np.rot90(Xb[i].transpose(1, 2, 0), r).transpose(2, 0, 1)

        if labels is not None:
            if not REGRESSION:
                labels = labels.astype(np.int32).ravel()
            else:
                labels = labels[:, np.newaxis]

        return Xb, labels

class DoubleIterator(QueueIterator):
    def __init__(self, mean, deterministic=False, *args, **kwargs):
        self.mean = mean
        self.deterministic = deterministic
        super(DoubleIterator, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        left_files, labels = super(QueueIterator, self).transform(Xb, yb)
        right_files = [f.replace('left', 'right') for f in left_files]

        Xb_l = (util.load_image(left_files) - self.mean) / MAX_PIXEL_VALUE
        Xb_r = (util.load_image(right_files) - self.mean) / MAX_PIXEL_VALUE

        # stack along x-axis of pixels
        Xb = np.concatenate([Xb_l, Xb_r], axis=2)

        return Xb, labels


class EarlyStopping(object):
    def __init__(self, patience=PATIENCE, loss='valid_loss',
                 greater_is_better=True):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.loss = loss
        self.greater_is_better = greater_is_better

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.loss] \
            * (-1.0 if self.greater_is_better else 1.0)
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
            nn.save_weights_to(WEIGHTS)
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

