import multiprocessing
import threading
import Queue
import time
from uuid import uuid4
import warnings

import numpy as np
#from nolearn.lasagne import BatchIterator

import SharedArray
from joblib import Parallel, delayed

import augment
import util

from definitions import *

N_PROC = 8

# TODO: try threading packend with parallel, might be faster

manager = multiprocessing.Manager()
shared_dict = manager.dict()


def load_image(args):
    i, fname, kwargs = args
    shared_dict[i] = augment.load(fname, **kwargs)


def load_shared(args):
    i, array_name, fname, kwargs = args
    array = SharedArray.attach(array_name)
    array[i] = augment.load(fname, **kwargs)


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

class BatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None, transform=None, color_vec=None):
        self.tf = transform
        self.color_vec = color_vec
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


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


class ThreadIterator(QueueIterator):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        super(QueueIterator, self).__init__(*args, **kwargs)

    def transform(self, Xb, yb, transform=None):
        fnames, labels = super(ThreadIterator, self).transform(Xb, yb)
        bs = len(Xb)
        Xb = np.zeros([bs, 3, self.model.cnf['w'], self.model.cnf['h']], 
                      dtype=np.float32)
        args = []
        for i, fname in enumerate(fnames):
            kwargs = self.model.cnf.copy()
            kwargs['transform'] = self.tf
            kwargs.update(self._get_metata())
            args.append((fname, kwargs))

        imgs = Parallel(n_jobs=N_PROC, backend='threading')(
                delayed(augment.load)(fname, **kwargs) for fname, kwargs in
                                                       args)
        
        for i, img in enumerate(imgs):
            Xb[i, ...] = img
            #Xs.append(augment.load(fname, **kwargs))

        #Xb = np.array(Xs)
        return Xb, labels


class SharedIterator(QueueIterator):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.pool = multiprocessing.Pool()
        super(SharedIterator, self).__init__(*args, **kwargs)

    def _get_metata(self):
        raise NotImplementedError


    def transform(self, Xb, yb):

        shared_array_name = str(uuid4())
        try:
            shared_array = SharedArray.create(
                shared_array_name, [len(Xb), 3, self.model.get('w'), 
                                    self.model.get('h')], dtype=np.float32)
                                        
            fnames, labels = super(SharedIterator, self).transform(Xb, yb)
            args = []

            for i, fname in enumerate(fnames):
                kwargs = {k: self.model.get(k) for k in 
                          ['w', 'h', 'rotate', 'aug_params', 'color', 'sigma', 
                           'mean', 'std']}
                kwargs['transform'] = getattr(self, 'tf', None)
                kwargs['color_vec'] = getattr(self, 'color_vec', None)
                kwargs.update(self._get_metata())
                args.append((i, shared_array_name, fname, kwargs))

            self.pool.map(load_shared, args)
            Xb = np.array(shared_array, dtype=np.float32)

        finally:
            SharedArray.delete(shared_array_name)

        return Xb, labels


class ManagerIterator(QueueIterator):
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.pool = multiprocessing.Pool()
        super(ManagerIterator, self).__init__(*args, **kwargs)

    def _get_metata(self):
        raise NotImplementedError

    def transform(self, Xb, yb):

        fnames, labels = super(ManagerIterator, self).transform(Xb, yb)
        args = []

        for i, fname in enumerate(fnames):
            kwargs = self.model.cnf.copy()
            kwargs['transform'] = transform
            kwargs.update(self._get_metata())
            args.append((i, fname, kwargs))
        
        self.pool.map(load_image, args)
        Xb = np.array([shared_dict[i] for i, _ in enumerate(fnames)])

        return Xb, labels


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

    def transform(self, Xb, yb, transform=None):
        fnames, labels = super(ProcessIterator, self).transform(Xb, yb)
        for i, fname in enumerate(fnames):

            kwargs = self.model.cnf.copy()
            kwargs['transform'] = transform
            kwargs.update(self._get_metata())

            self.tasks.put((augment.load, i, fname, kwargs))

        results = sorted([self.results.get() for _ in fnames])
        Xb = np.array([x for _, x in results], dtype=np.float32)
        return Xb, labels


class SingleIterator(SharedIterator):
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
    def __call__(self, X, y=None, transform=None, color_vec=None):

        #balance = max(self.model.get('balance', BALANCE_WEIGHT),
        #              self.model.get('min_balance', 0.0))
        #class_weights = (1.0 - balance) + balance \
        #        * np.array(self.model.get('class_weights', CLASS_WEIGHTS))
        #class_weights = self.model.get('balance_weights')
        #self.model.cnf['balance_weights'] *= self.model.cnf['balance_ratio']
        #class_weights = self.model.cnf['balance_weights'] \
        #        + self.model.cnf['final_balance_weights']
        alpha = self.model.cnf['balance_ratio'] ** self.count
        class_weights = self.model.cnf['balance_weights'] * alpha \
            + self.model.cnf['final_balance_weights'] * (1 - alpha)

        self.count += 1

        if y is not None and self.resample:
            n = len(y)
            indices = util.balance_per_class_indices(y, weights=class_weights)
            X = X[indices]
            y = y[indices]
        
            #self.model.cnf['balance'] = balance \
            #    * self.model.cnf.get('balance_ratio', 1)

        return super(SingleIterator, self).__call__(X, y, transform=transform,
                                                    color_vec=color_vec)

    def transform(self, Xb, yb, transform=None):
        Xb, labels = super(SingleIterator, self).transform(Xb, yb)

        # add dummy data for nervana kernels that need batch_size % 8 = 0
        missing = self.batch_size - len(Xb)
        tiles = np.ceil(float(missing) / len(Xb)).astype(int) + 1
        if missing != 0 and labels is not None:
            Xb = np.tile(Xb, [tiles] + [1] * (Xb.ndim - 1))[:self.batch_size]
            labels = np.tile(labels, [tiles] + [1] * (labels.ndim - 1))\
                [:self.batch_size]

        if labels is not None:
            if not self.model.get('regression', REGRESSION):
                labels = labels.astype(np.int32).ravel()
            else:
                labels = labels[:, np.newaxis]

        return Xb, labels

