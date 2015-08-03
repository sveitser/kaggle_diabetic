import multiprocessing
import threading
import Queue
from uuid import uuid4

import numpy as np
import SharedArray

import data


def load_shared(args):
    i, array_name, fname, kwargs = args
    array = SharedArray.attach(array_name)
    array[i] = data.load_augment(fname, **kwargs)


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


class SharedIterator(QueueIterator):
    def __init__(self, config, deterministic=False, *args, **kwargs):
        self.config = config
        self.deterministic = deterministic
        self.pool = multiprocessing.Pool()
        super(SharedIterator, self).__init__(*args, **kwargs)


    def transform(self, Xb, yb):

        shared_array_name = str(uuid4())
        try:
            shared_array = SharedArray.create(
                shared_array_name, [len(Xb), 3, self.config.get('w'), 
                                    self.config.get('h')], dtype=np.float32)
                                        
            fnames, labels = super(SharedIterator, self).transform(Xb, yb)
            args = []

            for i, fname in enumerate(fnames):
                kwargs = {k: self.config.get(k) for k in ['w', 'h']}
                if not self.deterministic:
                    kwargs.update({k: self.config.get(k) 
                                   for k in ['aug_params', 'sigma']})
                kwargs['transform'] = getattr(self, 'tf', None)
                kwargs['color_vec'] = getattr(self, 'color_vec', None)
                args.append((i, shared_array_name, fname, kwargs))

            self.pool.map(load_shared, args)
            Xb = np.array(shared_array, dtype=np.float32)

        finally:
            SharedArray.delete(shared_array_name)

        if labels is not None:
            labels = labels[:, np.newaxis]

        return Xb, labels


class ResampleIterator(SharedIterator):
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.count = 0
        super(ResampleIterator, self).__init__(config, *args, **kwargs)

    def __call__(self, X, y=None, transform=None, color_vec=None):
        if y is not None:
            alpha = self.config.cnf['balance_ratio'] ** self.count
            class_weights = self.config.cnf['balance_weights'] * alpha \
                + self.config.cnf['final_balance_weights'] * (1 - alpha)
            self.count += 1
            indices = data.balance_per_class_indices(y, weights=class_weights)
            X = X[indices]
            y = y[indices]
        return super(ResampleIterator, self).__call__(X, y, transform=transform,
                                                      color_vec=color_vec)

