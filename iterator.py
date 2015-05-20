import multiprocessing
import threading
import Queue
import time

import numpy as np
from nolearn.lasagne import BatchIterator

import augment
import util

from definitions import *

N_PROC = 8

class Consumer(multiprocessing.Process):
    
    def __init__(self, method, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.method = method
        self.daemon = True

    def run(self):
        try:
            while True:

                fun, id_, fname, kwargs = self.task_queue.get()
                img = fun(fname, **kwargs)
                self.result_queue.put((id_, img))

        except KeyboardInterrupt:
            return


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



class ProcessIterator(QueueIterator):
    def __init__(self, model, *args, **kwargs):
        self.tasks = multiprocessing.Queue()
        self.results = multiprocessing.Queue()
        self.model = model
        self.consumers = [Consumer(model.load, self.tasks, self.results) 
                          for _ in range(N_PROC)]
        for consumer in self.consumers:
            consumer.start()
        super(ProcessIterator, self).__init__(*args, **kwargs)

    def _get_metata(self):
        raise NotImplementedError

    def transform(self, Xb, yb):
        fnames, labels = super(ProcessIterator, self).transform(Xb, yb)
        for i, fname in enumerate(fnames):

            kwargs = self.model.cnf.copy()
            kwargs.update(self._get_metata())

            self.tasks.put((augment.load, i, fname, kwargs))

        results = sorted([self.results.get() for _ in fnames])
        Xb = np.array([x for _, x in results], dtype=np.float32)
        return Xb, labels


class SingleIterator(ProcessIterator):
    def __init__(self, model, deterministic=False,
                 resample=False, *args, **kwargs):
        self.deterministic = deterministic
        self.resample = resample
        self.model = model
        self.count = 0
        super(SingleIterator, self).__init__(model, *args, **kwargs)

    def _get_metata(self):
        return {'deterministic': self.deterministic}

    # this isn't needed if weights are set via masked objective
    def __call__(self, X, y=None):
        # balance classes in dataset
        if y is not None and self.resample:
            n = len(y)
            indices = util.balance_shuffle_indices(
                    y, random_state=None, 
                    weight=self.model.get('balance', BALANCE_WEIGHT))
            X = X[indices[:n]]
            y = y[indices[:n]]
        return super(SingleIterator, self).__call__(X, y)

    def transform(self, Xb, yb):
        Xb, labels = super(SingleIterator, self).transform(Xb, yb)

        # add dummy data for nervana kernels that need batch_size % 8 = 0
        missing = self.batch_size - len(Xb)
        tiles = np.ceil(float(missing) / len(Xb)).astype(int) + 1
        if missing != 0:
            Xb = np.tile(Xb, [tiles] + [1] * (Xb.ndim - 1))[:self.batch_size]
            if labels is not None:
                labels = np.tile(labels, [tiles] + [1] * (labels.ndim - 1))\
                        [:self.batch_size]

        if labels is not None:
            if not self.model.get('regression', REGRESSION):
                labels = labels.astype(np.int32).ravel()
            else:
                labels = labels[:, np.newaxis]

        return Xb, labels

