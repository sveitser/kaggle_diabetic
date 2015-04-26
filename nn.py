import Queue
import threading

import pandas as pd
import numpy as np

import lasagne
import lasagne.layers
from lasagne import init

from lasagne.updates import nesterov_momentum
from lasagne import updates
from nolearn.lasagne import NeuralNet, BatchIterator
import theano

from definitions import *
from layers import get_nn_layers
from util import load_images

def create_net(mean):
    net = NeuralNet(
        layers=get_nn_layers(),
        input_shape=(None, C, W, H),
        batch_iterator_train=FlipBatchIterator(batch_size=BATCH_SIZE,
                                               mean=mean),
        batch_iterator_test=FlipBatchIterator(batch_size=BATCH_SIZE,
                                              mean=mean),
        update=updates.nesterov_momentum,
        update_learning_rate=theano.shared(float32(INITIAL_LEARNING_RATE)),
        update_momentum=theano.shared(float32(INITIAL_MOMENTUM)),
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=INITIAL_LEARNING_RATE,
                           stop=0.0001),
            AdjustVariable('update_momentum', start=INITIAL_MOMENTUM,
                            stop=0.999),
            EarlyStopping(),
        ],
        use_label_encoder=False,
        eval_size=0.1,
        regression=True,
        max_epochs=MAX_ITER,
        verbose=2,
    )
    return net


def float32(k):
    return np.cast['float32'](k)


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
        queue = Queue.Queue(maxsize=10)
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


class FlipBatchIterator(QueueIterator):
    """BatchIterator with flipping and mean subtraction.

    Parameters
    ----------
    mean: np.array, dtype=np.float32
        with shape Channels x Width x Height

    batch_size: int
    """
    def __init__(self, mean, *args, **kwargs):
        self.mean = mean
        super(FlipBatchIterator, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb):
        files, labels = super(FlipBatchIterator, self).transform(Xb, yb)

        # Doing the type conversion here might take quite a lot of time
        # we could save some by preparing the data as numpy arrays
        # of np.float32 directly.
        Xb = load_images(files).astype(np.float32) - self.mean

        # bring values in range of [-0.5, 0.5]
        Xb /= MAX_PIXEL_VALUE

        # Flip half of the images in this batch at random in both dimensions
        bs = Xb.shape[0]
        
        # skip incomplete batches (use if some layers can't handle it)
        #if bs != BATCH_SIZE:
        #    raise StopIteration

        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, ::-1, :]

        #rotations = np.random.randint(4, size=bs)
        #for i, r in enumerate(rotations):
        #    Xb[i] = np.rot90(Xb[i].transpose(1, 2, 0), r).transpose(2, 0, 1)

        return Xb, labels[:, np.newaxis] if labels is not None else None


class EarlyStopping(object):
    def __init__(self, patience=20):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()
